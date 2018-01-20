"""This files reads the data and write it to features_x.csv"""
import csv
f = open("usage_data_day2.txt", "r")
data = []
for line in f:
    data.append(line.split())

tm = []
st = 0
last_hour = -1
for i in range(0, len(data)):
    tm = data[i][0].split(":")
    hour = int(tm[0])
    if last_hour != -1 and hour != last_hour:
        st += 1
    last_hour = hour
    hour = st
    val = ["19:01:2018:" + str(hour) + ":" + tm[1] + ":" + tm[2], data[i][-1]]
    with open("features_x.csv", "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(val)