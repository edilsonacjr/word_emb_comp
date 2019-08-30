

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = [15, 10]
sns.set_style("dark")

"""
    #TODO Questions:
        - How many papers in a year?
        - 
    
    
"""

df = pd.read_feather('data/all_data.feather')



fig, ax = plt.subplots()
df.hist('year',bins=40, ax=ax)
fig.savefig('plots/histogram.png')


bins = [0, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
df['binned_year'] = pd.cut(df['year'], bins=bins)

fig, ax = plt.subplots()
df.binned_year.value_counts(sort=False).plot.barh(ax=ax)
fig.suptitle('10 Year Groups', fontsize=20)
fig.savefig('plots/binned_bar.png')