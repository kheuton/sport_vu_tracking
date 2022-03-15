import csv
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# Feel free to add anything else you need here
import pandas as pd

# Read the event log into an array of dictionaries
events = []

metrics = pd.read_csv('final_metrics.csv')

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
