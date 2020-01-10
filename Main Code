import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

def plot_graphs(groundtruth,prediction,title):        
    N = 9
    ind = np.arange(N)  
    width = 0.40       

    fig = plt.figure()
    fig.suptitle(title, fontsize=10)
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, groundtruth, width, color='r')
    rects2 = ax.bar(ind+width, prediction, width, color='g')

    ax.set_ylabel("Amount of rainfall")
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC') )
    ax.legend( (rects1[0], rects2[0]), ('Ground truth', 'Prediction') )

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')


    plt.show()

district = pd.read_csv("district wise rainfall normal.csv",sep=",")
district = district.fillna(district.mean())
district.head()
ap_data = district[district['STATE_UT_NAME'] == 'TAMIL NADU']
ap_data[['DISTRICT', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("DISTRICT").mean()[:40].plot.barh(stacked=True,figsize=(18,8));
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

division_data = np.asarray(district[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])

X = None; y = None
for i in range(division_data.shape[1]-3):
    if X is None:
        X = division_data[:, i:i+3]
        y = division_data[:, i+3]
    else:
        X = np.concatenate((X, division_data[:, i:i+3]), axis=0)
        y = np.concatenate((y, division_data[:, i+3]), axis=0)
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
temp = district[['DISTRICT','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[district['STATE_UT_NAME'] == 'TAMIL NADU']
hyd = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['DISTRICT'] == 'CHENNAI'])
print(hyd); 
X_year = None; y_year = None
for i in range(hyd.shape[1]-3):
    if X_year is None:
        X_year = hyd[:, i:i+3]
        y_year = hyd[:, i+3]
    else:
        X_year = np.concatenate((X_year, hyd[:, i:i+3]), axis=0)
        y_year = np.concatenate((y_year, hyd[:, i+3]), axis=0)
Acclist=[];
scoretest=[]
for i in range(1,25):
    knn = KNeighborsRegressor(n_neighbors = i)
    knn.fit(X_train, y_train)
    scoretest.append(knn.score(X_test, y_test))

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
rf.fit(X_train, y_train)
print('Training accuracy: {:.3f}'.format(rf.score(X_train, y_train)*100))
print('Test accuracy: {:.3f}'.format(rf.score(X_test, y_test)*100))
p = max(scoretest)*100
p1 = round(p,2)
Acclist.append(p1)
y_pred = rf.predict(X_test)
y_year_pred = rf.predict(X_year)

plot_graphs(y_year,y_year_pred,"Prediction in Chennai")
rg = rf.predict([[12,43,64]])
print(rg);
print (rg/709);
