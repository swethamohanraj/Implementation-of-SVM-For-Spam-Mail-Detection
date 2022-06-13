# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4.Predict the required output.
5. End the program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by:K.M.SWETHA 
RegisterNumber: 212221240055 

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Data Head:
![image](https://user-images.githubusercontent.com/94228215/173282598-1e21b059-1bbe-46fc-b18b-411cdf4cd443.png)


### Data Info:
![image](https://user-images.githubusercontent.com/94228215/173282615-fdc41679-2da7-4757-a970-2931bcb295d8.png)


### Data isnull():
![image](https://user-images.githubusercontent.com/94228215/173282648-0446783c-06bd-4c12-a581-3eb7f78f72f6.png)


### y_pred:
![image](https://user-images.githubusercontent.com/94228215/173282671-584d38b3-8e47-4ce9-adc0-df8e8d312f50.png)


### Accuracy:
![image](https://user-images.githubusercontent.com/94228215/173282689-2d49d802-b9d9-4949-ba6f-bf345959b742.png)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
