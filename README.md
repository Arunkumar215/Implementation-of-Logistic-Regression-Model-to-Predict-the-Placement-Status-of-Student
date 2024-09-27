# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Arunkumar S A
RegisterNumber:212223220009  
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
1.PLACEMENT DATA


![image](https://github.com/user-attachments/assets/3229c970-0943-4cd1-ad27-d6aebfdd7057)


2.SALARY DATA:


![image](https://github.com/user-attachments/assets/c365312b-6507-4c8d-b193-a33151f78159)


3.CHECKING THE NULL() FUNCTION:


![image](https://github.com/user-attachments/assets/87f63f19-896a-44d3-b3a4-7e61eb0d8cca)


4.DATA DUPLICATE:


![image](https://github.com/user-attachments/assets/c458f119-e55a-49e6-9687-a9bce5bd03fe)


5.PRINT DATA:


![image](https://github.com/user-attachments/assets/732b5005-84e2-4a66-a4a7-9925c7732d20)


6.DATA STATUS:


![image](https://github.com/user-attachments/assets/a1b827c8-9281-447f-bf6f-54c1760d6498)


![image](https://github.com/user-attachments/assets/6be6bd53-c338-473b-bb02-9dd23f043e98)


7.Y_PREDICATION ARRAY:


![image](https://github.com/user-attachments/assets/4ebc89fc-51c7-4a4b-8f5a-69dda9c2dc50)


8.ACCURACY VALUE:


![image](https://github.com/user-attachments/assets/6db44496-f768-4e24-93a9-c97f77318191)


9.CONFUSION ARRAY:


![image](https://github.com/user-attachments/assets/f6ffef7d-2ded-4a23-a5e2-3d2bf092de7f)


![image](https://github.com/user-attachments/assets/6a85a147-51b0-449c-b7c5-6b435269ad3c)


10.CLASSIFICATION REPORT:


![image](https://github.com/user-attachments/assets/37ba3487-d8a8-416b-b764-95e7e952a9ed)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
