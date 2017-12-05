import numpy as np
import csv
import time
import re
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import math


filename = "data/GlobalTemperatures.csv"
raw_data = open(filename, 'r')
reader = csv.reader(raw_data)
x = list(reader)
tempx = x
print(np.shape(x))

fulldata = {}
date1 = "1849-12-31"

#getting the average by month
for y in x[1:]:
    date2 = y[0]
    if time.strptime(date1, "%Y-%m-%d") < time.strptime(date2, "%Y-%m-%d"):
        regex = re.compile('-\d{2}-')
        month = regex.search(date2)
        regex = re.compile('\d{2}')
        month = regex.search(month.group())
        y[0] = month.group()
        if(month.group() not in fulldata.keys()):
            fulldata[month.group()] = float(y[1])
        else:
            fulldata[month.group()] = fulldata[month.group()] + float(y[1])

print(fulldata)
number_of_years = 2013 - 1850

for key in fulldata.keys():
    fulldata[key] = fulldata[key]/number_of_years

print(fulldata)
lists = sorted(fulldata.items())
x, y = zip(*lists)
plt.plot(x, y)
# plt.show()


filename = "data/GlobalTemperatures.csv"
raw_data = open(filename, 'r')
reader = csv.reader(raw_data)
x = list(reader)

print(x)

x = np.delete(x,2,axis = 1)
x = np.delete(x,3,axis = 1)
x = np.delete(x,4,axis = 1)
x = np.delete(x,5,axis = 1)

print(x[0])
print(np.shape(x))

z = np.zeros((3193,1), dtype=int)
x = np.append(x,z,axis=1)
print(x[0])

x = np.ndarray.tolist(x)
newdataset = []
regex = re.compile('\d{4}')
date1 = "1849-12-31"
year1 = regex.search(date1)
print('year1',year1.group())
for y in x[1:]:
    date2 = y[0]
    if time.strptime(date1, "%Y-%m-%d") < time.strptime(date2, "%Y-%m-%d"):
        regex = re.compile('-\d{2}-')
        month = regex.search(date2)
        regex = re.compile('\d{2}')
        month = regex.search(month.group())
        y[0] = month.group()
        newdataset.append(y)

print(newdataset[0])

for row in newdataset:
    temp1 = float(row[1]) + float(row[1]) * 0.05
    temp2 = float(row[1]) - float(row[1]) * 0.05
    if(math.fabs( temp1 > fulldata[row[0]] and fulldata[row[0]] > temp2)):
        row[5] = 0
    else:
        row[5] = 1

print(newdataset)
print(fulldata['01'])

newdataset = np.delete(newdataset,0,axis=1)
print(newdataset[0])
newdataset = np.delete(newdataset,0,axis=1)
print(newdataset[0])
newdataset = np.asarray(newdataset,dtype=np.float64)
labels = np.asarray(newdataset)[:,3]
labels = np.ndarray.tolist(labels)
print(labels[0])
newdataset = np.delete(newdataset,3,axis=1)
print(newdataset[0])
trainX = newdataset[:int((len(newdataset)+1)*.80)] #Remaining 80% to training set
testX = newdataset[int(len(newdataset)*.80+1):]
trainY =  labels[:int((len(labels)+1)*.80)]
testY = labels[int((len(labels)+1)*.80):]


model = LogisticRegression()
model.fit(trainX, trainY)
print('logistic score',model.score(trainX, trainY))
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
#Predict Output
predicted= model.predict(testX)
print('mse logistic',np.mean((predicted - testY) ** 2))

####
from sklearn import svm

model = svm.SVC() # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.
# Train the model using the training sets and check score
model.fit(trainX, trainY)
print('svm score',model.score(trainX, trainY))
#Predict Output
predicted= model.predict(testX)
print('mse logistic',np.mean((predicted - testY) ** 2))

####
from sklearn.ensemble import RandomForestClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
model= RandomForestClassifier()
# Train the model using the training sets and check score
model.fit(trainX,trainY)
print('random forest score',model.score(trainX,trainY))
#Predict Output
predicted= model.predict(testX)
print('mse logistic',np.mean((predicted - testY) ** 2))