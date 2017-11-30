import numpy as np
import csv
import time
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

filename = "data/GlobalTemperatures.csv"
raw_data = open(filename, 'r')
reader = csv.reader(raw_data)
x = list(reader)
print(x[0])

#remove the uncertainity columns
x = np.delete(x,2,axis = 1)
x = np.delete(x,3,axis = 1)
x = np.delete(x,4,axis = 1)
x = np.delete(x,5,axis = 1)

print(x[0])

# labels = x[:,[1]]
# labels = np.ndarray.tolist(labels)

x = np.ndarray.tolist(x)
fulldata = []
# fulldata.append(x[0])

date1 = "1849-12-31"

for y in x[1:]:
    date2 = y[0]
    if time.strptime(date1, "%Y-%m-%d") < time.strptime(date2, "%Y-%m-%d"):
        # regex = re.compile('\d{4}')
        # year = regex.search(date2)
        # # print('year',year.group())
        # y[0] = year.group()
        fulldata.append(y)

# fulldata = np.dtype('f8')
#remove the 'year' column

fulldata = np.delete(fulldata,0,axis=1)
fulldata = np.asarray(fulldata,dtype=np.float64)
fulldata = np.ndarray.tolist(fulldata)
print('fuldata=',fulldata[0])

#get the first column i.e. landaverage temprature as labels
labels = np.asarray(fulldata)[:,0]
labels = np.ndarray.tolist(labels)
print(labels[0])

#remove the colums used as labels
fulldata = np.delete(fulldata,0, axis = 1)
fulldata = np.ndarray.tolist(fulldata)
print(fulldata[0])
trainX = fulldata[:int((len(fulldata)+1)*.80)] #Remaining 80% to training set
testX = fulldata[int(len(fulldata)*.80+1):]

print(np.shape(trainX),np.shape(testX))
#
# labels = np.asarray(fulldata)[:,1]
# labels = np.ndarray.tolist(labels)

trainY =  labels[:int((len(labels)+1)*.80)]
testY = labels[int((len(labels)+1)*.80):]

print(np.shape(trainY),np.shape(testY))

model = linear_model.LinearRegression()
model.fit(trainX,trainY)

print('importance',model.coef_,'score',model.score(trainX,trainY))
predicted = model.predict(testX)

print(np.shape(predicted), np.shape(testY))
print(predicted[0],testY[0])
# print(np.mean((predicted - testY) ** 2))
sum = 0

for i in range(len(predicted)):
    sum+= (float(predicted[i]) - float(testY[i])) ** 2
import math
print(sum,sum/len(predicted),math.sqrt(sum/len(predicted)))






# x = np.delete(x,2,axis = 1)
# x = np.delete(x,3,axis = 1)
# x = np.ndarray.tolist(x)
# print(x[0])
# state = []
# date1 = "1950-12-31"
# state.append(x[0])
# for y in x[1:]:
#     date2 = y[0]
#     if time.strptime(date1, "%Y-%m-%d") < time.strptime(date2, "%Y-%m-%d") and y[2] == 'Virginia' :
#         state.append(y)
#
# print(state)
# print(len(state))
# print(len(state)/12)