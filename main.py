import numpy as np
import csv
import time

filename = "data/GlobalLandTemperaturesByState.csv"
raw_data = open(filename, 'r')
reader = csv.reader(raw_data)
x = list(reader)
print(x[0])
x = np.delete(x,2,axis = 1)
x = np.delete(x,3,axis = 1)
x = np.ndarray.tolist(x)
print(x[0])
state = []
date1 = "1950-12-31"
state.append(x[0])
for y in x[1:]:
    date2 = y[0]
    if time.strptime(date1, "%Y-%m-%d") < time.strptime(date2, "%Y-%m-%d") and y[2] == 'Virginia' :
        state.append(y)

print(state)
print(len(state))
print(len(state)/12)