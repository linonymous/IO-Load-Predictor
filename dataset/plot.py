import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
# datetime.datetime()

df = pd.read_csv("features_x.csv", delimiter=",", header=None)
print len(df.index)
date = []
idle = []
for i in range(len(df.index)):
    a = df.iloc[i, 0]
    lst = [int(x) for x in a.split(":")]
    date.append(datetime.datetime(lst[2], lst[1], lst[0], lst[3], lst[4], lst[5]))
    idle.append(df.iloc[i, 1])

y = np.array(idle)
"""Normalize the feature date"""
date_features = [(i - min(date)).total_seconds() for i in date]
date_features = [i/max(date_features) for i in date_features]
x = np.array(date_features)
plt.plot(x[:], y[:])
plt.show()
