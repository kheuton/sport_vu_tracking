import numpy as np
import pandas as pd

rim_1_x = 5.25
rim_1_y = 25
rim_2_x = 94-5.25
rim_2_y = 25

shot_radius = 6


def make_position_dataframe(event_data, target_team_id, target_player_id, target_event_id=None):
    event_df = pd.DataFrame()

    for event in event_data['events']:
        event_id = event['eventId']

        if target_event_id is not None and event_id != str(target_event_id):
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

                # if team_id ==target_team_id:
                # import pdb; pdb.set_trace()

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

    return event_df

def augment_data(ball_data):
    # Smooth pos
    ball_data['avg_x'] = ball_data.rolling(20, min_periods=1).mean()['x']
    ball_data['avg_y'] = ball_data.rolling(20, min_periods=1).mean()['y']
    ball_data['avg_z'] = ball_data.rolling(20, min_periods=1).mean()['z']

    # get dist to rims
    ball_data['dist_to_rim_1'] = np.sqrt(
        (ball_data['avg_x'] - rim_1_x) ** 2 + (ball_data['avg_y'] - rim_1_y) ** 2)
    ball_data['dist_to_rim_2'] = np.sqrt(
        (ball_data['avg_x'] - rim_2_x) ** 2 + (ball_data['avg_y'] - rim_2_y) ** 2)

    # When the ball drops from above 10 ft to below
    ball_data['10_ft_drops'] = ball_data['avg_z'].rolling(2, min_periods=2).apply(lambda x:
                                                                                  x.values[1] < 10 and x.values[0] > 10)

    # When the ball is close to a rim
    ball_data['close_to_rim_1'] = ball_data['dist_to_rim_1'] < shot_radius
    ball_data['close_to_rim_2'] = ball_data['dist_to_rim_2'] < shot_radius

    # if it drops and is close to a rim, that's a shot landing
    ball_data['shot_arrival_rim_1'] = ball_data['close_to_rim_1'] * ball_data['10_ft_drops']
    rim_1_shots = np.where(ball_data['shot_arrival_rim_1'] == 1)[0]
    ball_data['shot_arrival_rim_2'] = ball_data['close_to_rim_2'] * ball_data['10_ft_drops']
    rim_2_shots = np.where(ball_data['shot_arrival_rim_2'] == 1)[0]

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