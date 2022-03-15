import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers import make_position_dataframe, augment_data, get_shot_stats, find_closest, process_shots, rescale_metrics

# load data
with open('0021500495.json', 'r') as json_file:
    event_data = json.load(json_file)


event_ids = []
for event in event_data['events']:
    event_ids.append(event['eventId'])

max_observed_timestep = -1

shot_df = pd.DataFrame()
for event_id in event_ids:

    # Get dataframe of ball position data
    ball_data, event = make_position_dataframe(event_data, -1, -1, event_id)

    if len(ball_data) == 0:
        # print(f'No data {event_id}')
        continue

    min_data_timestep = ball_data['timestamp'].values.min()
    max_data_timestep = ball_data['timestamp'].values.max()

    if max_data_timestep <= max_observed_timestep:
        # print(f'Duplicate event {event_id}, skipping')
        continue
    # elif min_data_timestep <= max_observed_timestep:
    # print(f'WARNING: partially duplicate data {event_id}')

    # Add extra stats like distance to each rim, velocity/acceleration, shot arrivals
    ball_data = augment_data(ball_data)

    # Get shots at each rim
    rim_1_shots = np.where(ball_data['shot_arrival_rim_1'] == 1)[0]
    rim_2_shots = np.where(ball_data['shot_arrival_rim_2'] == 1)[0]

    # Calculate shooter and augment shot with custom quality metric
    shot_df = process_shots(shot_df, ball_data, event, event_data, event_id, max_observed_timestep,
                            1, rim_1_shots)
    shot_df = process_shots(shot_df, ball_data, event, event_data, event_id, max_observed_timestep,
                            2, rim_2_shots)

    max_observed_timestep = max_data_timestep

shot_df['min_rem'] = shot_df['qtr_rem'].apply(lambda x: np.floor(x/60))
shot_df['sec_rem'] = shot_df['qtr_rem'].apply(lambda x: x-60*np.floor(x/60))
shot_df['shot_times'] = 12*60-shot_df['qtr_rem'] + 12*60*(shot_df['qtr']-1)

# Scale quality 0-10
metrics = rescale_metrics(shot_df)

metrics.to_csv('final_metrics.csv',index=False)

# YOUR SOLUTION GOES HERE
# These are the two arrays that you need to populate with actual data
shot_times = metrics.shot_times.values # Between 0 and 2880
shot_facts = metrics.quality_score.values # Scaled between 0 and 10




# This code creates the timeline display from the shot_times
# and shot_facts arrays.
# DO NOT MODIFY THIS CODE APART FROM THE SHOT FACT LABEL
fig, ax = plt.subplots(figsize=(12,3))
fig.canvas.set_window_title('Shot Timeline')

plt.scatter(shot_times, np.full_like(shot_times, 0), marker='o', s=50, color='royalblue', edgecolors='black', zorder=3, label='shot')
plt.bar(shot_times, shot_facts, bottom=2, color='royalblue', edgecolor='black', width=5, label='Quality (Close to rim + Away from defender + Set feet)') # <- This is the label you can modify

ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.tick_params(axis='x', length=20)
ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0,720,1440,2160,2880]))
ax.set_yticks([])

_, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(-15, xmax)
ax.set_ylim(ymin, ymax+5)
ax.text(xmax, 2, "time", ha='right', va='top', size=10)
plt.legend(ncol=5, loc='upper left')

plt.tight_layout()
plt.show()

plt.savefig("Shot_Timeline.png")