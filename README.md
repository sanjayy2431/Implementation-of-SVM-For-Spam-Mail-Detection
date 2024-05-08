# Implementation-of-SVM-For-Spam-Mail-Detection
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
```
1.Import the required libraries.
2.Read the data frame using pandas.
3.Get the information regarding the null values present in the dataframe.
4.Split the data into training and testing sets.
5.Convert the text data into a numerical representation using CountVectorizer.
6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7.Finally, evaluate the accuracy of the model. 
```
## Program:

/* Program to implement the SVM For Spam Mail Detection    
Developed by: SANJAY V    
RegisterNumber: 212223230188   
*/

import chardet    
file='spam.csv'   
with open(file, 'rb') as rawdata:    
result = chardet.detect(rawdata.read(100000))    
result    
import pandas as pd     
data = pd.read_csv("spam.csv",encoding="Windows-1252")     
data.head()    
data.info()   
data.isnull().sum()   
X = data["v1"].values   
Y = data["v2"].values    
from sklearn.model_selection import train_test_split    
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)   
from sklearn.feature_extraction.text import CountVectorizer    
cv = CountVectorizer()    
X_train = cv.fit_transform(X_train)   
X_test = cv.transform(X_test)    
from sklearn.svm import SVC    
svc=SVC()   
svc.fit(X_train,Y_train)    
Y_pred = svc.predict(X_test)    
print("Y_prediction Value: ",Y_pred)   
from sklearn import metrics    
accuracy=metrics.accuracy_score(Y_test,Y_pred)    
accuracy    
Program to implement the SVM For Spam Mail Detection.    


## Output: 
## result output:   
![image](https://github.com/sanjayy2431/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365143/a83ef5c9-4219-45dd-b977-48fd46be7642)
## data.head()    
![image](https://github.com/sanjayy2431/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365143/65e15ea4-0948-4502-a9b0-25d3a238390b)                 
## data.info()     
![image](https://github.com/sanjayy2431/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365143/cbc130ba-8dce-4228-997c-58ca8906c4c2)
## data.isnull().sum()    
![image](https://github.com/sanjayy2431/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365143/56a0cf18-6817-42e5-95c4-19f97afea382)
![image](https://github.com/sanjayy2431/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365143/7906518d-655f-45df-9c02-5d4102f596c5)
## Y_prediction Value    
![image](https://github.com/sanjayy2431/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365143/3d887584-92f3-43b4-84f5-d7867e16e0f9)
## Accuracy Value    
![image](https://github.com/sanjayy2431/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149365143/28241601-5b4f-4d6c-96ea-f53202004fac)   
## Result:  
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
