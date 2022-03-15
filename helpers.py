import math
import numpy as np
import pandas as pd

rim_1_x = 5.25
rim_1_y = 25
rim_2_x = 94-5.25
rim_2_y = 25

shot_radius = 6

# rim[qtr:team]
# We learned this from tracking data, we didn't cheat
which_side_shoots = {1:{1:1610612738, 2:1610612738, 3:1610612751, 4:1610612751},
                     2:{3:1610612738, 4:1610612738, 1:1610612751, 2:1610612751}}


def make_position_dataframe(event_data, target_team_id, target_player_id, target_event_id):
    """Create a pandas dataframe from the JSON data for a specific event and player

    Args:
        event_data: The json data of the entire game
        target_team_id: Team of our target player
        target_player_id: Player we want
        target_event_id: Event we want
    """
    event_df = pd.DataFrame()

    for event in event_data['events']:
        event_id = event['eventId']

        if event_id != str(target_event_id):
            continue

        moment_df = pd.DataFrame()

        for moment in event['moments']:
            timestamp = moment[1]
            qtr = moment[0]
            qtr_rem = moment[2]
            shot_rem = moment[3]

            for player_pos_data in moment[5]:
                team_id = player_pos_data[0]
                player_id = player_pos_data[1]

                if team_id == target_team_id and player_id == target_player_id:
                    x = player_pos_data[2]
                    y = player_pos_data[3]
                    z = player_pos_data[4]

                    row = {'eventId': event_id, 'timestamp': timestamp,
                           'qtr': qtr, 'qtr_rem': qtr_rem, 'shot_rem': shot_rem,
                           'team_id': team_id, 'player_id': player_id,
                           'x': x, 'y': y, 'z': z}

                    moment_df = pd.concat([moment_df, pd.DataFrame([row])])

        event_df = pd.concat([event_df, moment_df])

        return event_df, event


def rescale_metrics(shot_df):
    """Rescale our quality components 0-1, and scale composite 0-10"""
    result_df = shot_df.copy()

    # Rescale components to be 1
    min_vel, max_vel = result_df['vel_shooter'].min(), result_df['vel_shooter'].max()
    min_def, max_def = result_df['closest_def'].min(), result_df['closest_def'].max()
    min_dist, max_dist = result_df['shot_dist'].min(), result_df['shot_dist'].max()

    # 0 velocity is good
    result_df['scaled_vel_shooter'] = 1 - (result_df['vel_shooter'] - min_vel) / (max_vel - min_vel)
    # large distance is good
    result_df['scaled_closest_def'] = (result_df['closest_def'] - min_def) / (max_def - min_def)
    # 0 distance is good
    result_df['scaled_shot_dist'] = 1 - (result_df['shot_dist'] - min_dist) / (max_dist - min_dist)

    result_df['quality_score'] = result_df['scaled_vel_shooter'] + result_df['scaled_closest_def'] + result_df[
        'scaled_shot_dist']

    # rescale to 0-10
    min_qual, max_qual = result_df['quality_score'].min(), result_df['quality_score'].max()

    result_df['quality_score'] = 10 * (result_df['quality_score'] - min_qual) / (max_qual - min_qual)

    return result_df


def augment_data(ball_data):
    """Add distance, velocity, and acceration information to each rim"""

    # Smooth positon
    ball_data['avg_x'] = ball_data.rolling(20, min_periods=1).mean()['x']
    ball_data['avg_y'] = ball_data.rolling(20, min_periods=1).mean()['y']
    ball_data['avg_z'] = ball_data.rolling(20, min_periods=1).mean()['z']

    # get dist to rims
    ball_data['dist_to_rim_1'] = np.sqrt(
        (ball_data['avg_x'] - rim_1_x) ** 2 + (ball_data['avg_y'] - rim_1_y) ** 2)
    ball_data['dist_to_rim_2'] = np.sqrt(
        (ball_data['avg_x'] - rim_2_x) ** 2 + (ball_data['avg_y'] - rim_2_y) ** 2)

    # When the ball is descending from above the rim
    ball_data['dropping_above_rim'] = ball_data['avg_z'].rolling(2, min_periods=2).apply(lambda x:
        (x.values[1] <  x.values[0]) and (x.values[1] >= 10.0))

    # first 1 is NA
    ball_data['dropping_above_rim'] = ball_data['dropping_above_rim'].fillna(0)

    # When the ball is close to a rim
    ball_data['close_to_rim_1'] = ball_data['dist_to_rim_1'] < shot_radius
    ball_data['close_to_rim_2'] = ball_data['dist_to_rim_2'] < shot_radius

    # if it drops and is close to a rim, that's a shot landing
    ball_data['shot_arrival_rim_1'] = ball_data['close_to_rim_1'] * ball_data['dropping_above_rim']
    ball_data['shot_arrival_rim_2'] = ball_data['close_to_rim_2'] * ball_data['dropping_above_rim']

    # We'll have a huge stretch of 1's every frame the ball falls while above the rim
    # Just keep the last one.
    ball_data['shot_arrival_rim_1'] = (ball_data['shot_arrival_rim_1'].shift(-1) != ball_data['shot_arrival_rim_1']) * \
                                      ball_data['shot_arrival_rim_1']
    ball_data['shot_arrival_rim_2'] = (ball_data['shot_arrival_rim_2'].shift(-1) != ball_data['shot_arrival_rim_2']) * \
                                      ball_data['shot_arrival_rim_2']


    # get velocity/acceleration
    ball_data['vel_rim_1'] = -ball_data['dist_to_rim_1'].diff()
    ball_data['avg_vel_rim_1'] = ball_data.rolling(5, min_periods=1).mean()['vel_rim_1']
    ball_data['vel_rim_2'] = -ball_data['dist_to_rim_2'].diff()
    ball_data['avg_vel_rim_2'] = ball_data.rolling(5, min_periods=1).mean()['vel_rim_2']
    ball_data['acc_rim_1'] = ball_data['vel_rim_1'].diff()
    ball_data['avg_acc_rim_1'] = ball_data.rolling(20, min_periods=1).mean()['acc_rim_1']
    ball_data['acc_rim_2'] = ball_data['vel_rim_2'].diff()
    ball_data['avg_acc_rim_2'] = ball_data.rolling(5, min_periods=1).mean()['acc_rim_2']

    # mark where a ball starts accelrating towards a rim
    ball_data['go_to_rim_1'] = ball_data['avg_acc_rim_1'].rolling(2, min_periods=2).apply(lambda x:
                                                                                          x.values[1] > 0 and x.values[
                                                                                              0] < 0)
    ball_data['go_to_rim_2'] = ball_data['avg_acc_rim_2'].rolling(2, min_periods=2).apply(lambda x:
                                                                                          x.values[1] > 0 and x.values[
                                                                                              0] < 0)

    return ball_data


def find_closest(moment, close_type='player', shooting_team=None):
    """Find the closest player or team to the ball for a single moment

    Args:
        moment: Single moment from a specific event in the event json
        close_type: Will return the closest team to the ball, or player to the ball
        shooting_team: Optional, if provided, will exclude players of the opposing team when calculating closest.
            Used to find the shooter.
    """
    # Get coordinates of the ball
    ball_x = moment[5][0][2]
    ball_y = moment[5][0][3]

    # Make a list of all players distance to the ball
    ball_dist = np.zeros(10)
    for i in range(1, 11):
        player_x = moment[5][i][2]
        player_y = moment[5][i][3]
        player_team = moment[5][i][0]
        ball_dist[i - 1] = math.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)

        # Only consider shooting team if provided
        if shooting_team is not None:
            if player_team != shooting_team:
                ball_dist[i - 1] = np.inf

    # finds the minimum distance and which person has that distance
    min_value = np.amin(ball_dist)
    min_index = np.where(ball_dist == min_value)

    # Get the index of out the double array
    index = min_index[0][0]

    if close_type == 'team':
        close_id = moment[5][index + 1][0]
    elif close_type == 'player':
        close_id = moment[5][index + 1][1]

    return close_id


def get_shot_stats(event_data, event_id, shot_idx, shooter, shooter_team, rim):
    """Add our custom quality components, distance, shooter velocity, and defense coverage"""
    shooter_data, event = make_position_dataframe(event_data, shooter_team, shooter, event_id)

    shooter_data['x_dist'] = shooter_data['x'].rolling(2, min_periods=2).apply(lambda x:
        (x.values[1] - x.values[0])**2).fillna(0)
    shooter_data['y_dist'] = shooter_data['x'].rolling(2, min_periods=2).apply(lambda x:
                                                                               (x.values[1] - x.values[0]) ** 2).fillna(0)
    shooter_data['delta_dist'] = np.sqrt(shooter_data['x_dist'] + shooter_data['y_dist'])
    shooter_data['velocity'] = shooter_data.rolling(5, min_periods=1).mean()['delta_dist'].fillna(0)

    shooter_velocity = shooter_data['velocity'].values[shot_idx]

    shot_moment = event['moments'][shot_idx]

    closest_defender_dist = 6

    if rim==1:
        rim_x = rim_1_x
        rim_y = rim_2_y
    else:
        rim_x = rim_2_x
        rim_y = rim_2_y

    shooter_dist_to_rim = np.sqrt((shooter_data['x'].values[shot_idx] - rim_x) ** 2 +
                                  (shooter_data['y'].values[shot_idx] - rim_y) ** 2)

    for defender in shot_moment[5]:
        defender_team = defender[0]
        if defender_team==-1 or defender_team==shooter_team:
            # not a defender
            continue

        def_dist_to_rim = np.sqrt((defender[2]-rim_x)**2+(defender[3]-rim_y)**2)
        # Defender is between shooter and ball
        if def_dist_to_rim < shooter_dist_to_rim:
            def_dist_to_shooter = np.sqrt((defender[2]-shooter_data['x'].values[shot_idx])**2+
                                          (defender[3]-shooter_data['y'].values[shot_idx])**2)
            if def_dist_to_shooter <closest_defender_dist:
                closest_defender_dist=def_dist_to_shooter



    return shooter_velocity, closest_defender_dist, shooter_dist_to_rim

def process_shots(shot_df, ball_data, event, event_data, event_id, max_observed_timestep, rim, shot_indices):
    """Given a list of moments when shots occured, calculate all relevant stats"""

    rim = int(rim)

    for shot in shot_indices:
        rim_accels = np.where(ball_data[f'go_to_rim_{rim}'].values == 1)[0]

        if len(rim_accels[rim_accels < shot]) == 0:
            print(f'Ball never accelerated towards rim! {event_id}')
            print(f'Assuming event started late. Shot at {shot}')
            rim_accels = np.append([0], rim_accels)

        shot_index = max(rim_accels[rim_accels < shot])

        time_hit_rim = ball_data['timestamp'].values[shot]

        shot_moment = event['moments'][shot_index]
        qtr = shot_moment[0]
        shooting_team = which_side_shoots[rim][qtr]
        closest_team = find_closest(shot_moment, close_type='team', shooting_team=shooting_team)
        assert (closest_team == shooting_team)
        closest_player = find_closest(shot_moment, close_type='player', shooting_team=shooting_team)

        vel_shooter, closest_def, shot_dist = get_shot_stats(event_data, event_id, shot_index,
                                                             closest_player, shooting_team, 1)

        if time_hit_rim > max_observed_timestep:
            ball_row = ball_data.iloc[shot_index]

            shot_row = ball_row[['eventId', 'qtr', 'qtr_rem', 'timestamp', 'x', 'y', 'z']].append(
                pd.Series({'rim': rim, 'closest_team': closest_team, 'shot_idx': shot_index,
                           'shooter': closest_player, 'shooting_team': shooting_team,
                           'vel_shooter': vel_shooter, 'closest_def': closest_def, 'shot_dist': shot_dist}))
            shot_df = shot_df.append(shot_row, ignore_index=True)
        # else:
        # print(f'oh no, this shot occurs before something weve seen before: {event_id}')

    return shot_df

