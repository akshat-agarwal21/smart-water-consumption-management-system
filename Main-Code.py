# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import RPi.GPIO as GPIO 
import time, datetime, sys
from datetime import datetime
import seaborn as sns


pn = pd.read_csv("pop.csv")
cpn = pn[pn['State']=='TAMIL NADU']
tn = cpn[cpn['District']=='Chennai']
x=tn['Population'].values/10000


data = pd.read_csv("district wise rainfall normal.csv")
print(data.head())
def plot_graphs(groundtruth,prediction,title):        
    N = 9
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars

    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, groundtruth, width, color='r')
    rects2 = ax.bar(ind+width, prediction, width, color='g')

    ax.set_ylabel("Amount of rainfall")
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC') )
    ax.legend( (rects1[0], rects2[0]), ('Ground truth', 'Prediction') )

#     autolabel(rects1)
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')
#     autolabel(rects2)

    plt.show()

data[['DISTRICT', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("DISTRICT").mean()[:40].plot.barh(stacked=True,figsize=(13,8));
data[['DISTRICT', 'Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']].groupby("DISTRICT").sum()[:40].plot.barh(stacked=True,figsize=(16,8));
tn_data = data[data['STATE_UT_NAME'] == 'TAMIL NADU']
tn_data[['DISTRICT', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("DISTRICT").mean()[:40].plot.barh(stacked=True,figsize=(18,8));
tn_data[['DISTRICT', 'Jan-Feb', 'Mar-May',
       'Jun-Sep', 'Oct-Dec']].groupby("DISTRICT").sum()[:40].plot.barh(stacked=True,figsize=(16,8));
plt.figure(figsize=(11,4))
sns.heatmap(tn_data[['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec','ANNUAL']].corr(),annot=True)
plt.show()

plt.figure(figsize=(11,4))
sns.heatmap(tn_data[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']].corr(),annot=True)
plt.show()

Acclist = []
division_data = np.asarray(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])

x = None; y = None
for i in range(division_data.shape[1]-3):
    if x is None:
        x = division_data[:, i:i+3]
        y = division_data[:, i+3]
    else:
        x = np.concatenate((x, division_data[:, i:i+3]), axis=0)
        y = np.concatenate((y, division_data[:, i+3]), axis=0)
        
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
temp = data[['DISTRICT','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['STATE_UT_NAME'] == 'TAMIL NADU']
maa = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['DISTRICT'] == 'CHENNAI'])
# print temp
x_year = None; y_year = None
for i in range(maa.shape[1]-3):
    if x_year is None:
        x_year = maa[:, i:i+3]
        y_year = maa[:, i+3]
    else:
        x_year = np.concatenate((x_year, maa[:, i:i+3]), axis=0)
        y_year = np.concatenate((y_year, maa[:, i+3]), axis=0)
scoretest=[]
for i in range(1,25):
    knn = KNeighborsRegressor(n_neighbors = i)
    knn.fit(x_train, y_train)
    scoretest.append(knn.score(x_test, y_test))

plt.plot(range(1, 25), scoretest)
plt.xticks(np.arange(1, 25, 1))
plt.xlabel('K-value')
plt.ylabel('Accuracy')
plt.show()

print('Maximum KNN Accuracy is: {:.2f}%'.format((max(scoretest))*100))
p = max(scoretest)*100
p1 = round(p,2)
Acclist.append(p1)
rf = RandomForestRegressor(n_estimators = 20, random_state = 2)
rf.fit(x_train, y_train)
print('Training accuracy: {:.3f}'.format(rf.score(x_train, y_train)*100))
print('Test accuracy: {:.3f}'.format(rf.score(x_test, y_test)*100))
p = max(scoretest)*100
p1 = round(p,2)
Acclist.append(p1)


y_pred = rf.predict(x_test)
y_year_pred = rf.predict(x_year)

plot_graphs(y_year,y_year_pred,"Prediction in Chennai")



rg = (rf.predict([[12,43,64]]))*10000/tn['Population'].values
lphpd = (rg/0.5)*200
print("maximum water that can be used by a single user per day= ")
print(lphpd)
pphph = lphpd*7.5*60
print(pphph)





fg = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(fg, GPIO.IN, pull_up_down = GPIO.PUD_UP)
global count 
count =0
def countPulse(channel):
    global count 
    count+=1
    print(count)
    if count>=pphph:
         GPIO.setmode(GPIO.BCM)
         pinList = [14, 15, 18, 23]
         for i in pinList:
           GPIO.setup(i, GPIO.OUT)
           GPIO.output(i, GPIO.HIGH)
#time for which the valve remains open
         todays_date = datetime.now()

# to get hour from datetime
         l=todays_date.hour

# to get minute from datetime
         m=todays_date.minute
         n=todays_date.second
         total = 3600*l + 60*m +n
         
         print(total)
         SleepTimeL = 86400-total
# 10 is equivalent to 10 seconds 
         try:
           GPIO.output(14, GPIO.LOW) 
           time.sleep(SleepTimeL);
           GPIO.cleanup()
           print ("Good bye!")

         except KeyboardInterrupt:
           print ("  Quit")
         GPIO.cleanup()
   
GPIO.add_event_detect(fg, GPIO.FALLING, callback=countPulse)

while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        print("caught")
        GPIO.cleanup()
        sys.exit()


