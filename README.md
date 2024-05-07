# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.    
2.Read the data frame using pandas.    
3.Get the information regarding the null values present in the dataframe.    
4.Split the data into training and testing sets.    
5.Convert the text data into a numerical representation using CountVectorizer.     
6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.    
7.Finally, evaluate the accuracy of the model.

## Program:
/*
Program to implement the SVM For Spam Mail Detection.    
Developed by: SANJAY ASHWIN P    
RegisterNumber: 212223040181     
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



## Output:

### Result Output   
![image](https://github.com/sanjayashwinP/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473265/1d6152c3-5724-44c3-b02a-62dcb39a92a5)
### data.head()    
![image](https://github.com/sanjayashwinP/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473265/3ff23046-1d9c-46e5-98bd-32a9fe0a9bab)
### data.info()    
![image](https://github.com/sanjayashwinP/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473265/79c5aace-560b-4bc8-8721-668f6084446e)

### data.isnull().sum()    
![image](https://github.com/sanjayashwinP/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473265/cbf08416-c8f7-4de4-8658-0276c97d0526)
![image](https://github.com/sanjayashwinP/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473265/0a1217e0-fa3c-4f3f-bce8-bbf725dae9e1)

### Y_prediction Value    
![image](https://github.com/sanjayashwinP/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473265/8e300a9c-70c5-4b66-8524-672bb96b3bc1)

### Accuracy Value   
![image](https://github.com/sanjayashwinP/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473265/8956be96-b15f-422d-918b-3c77de4f8b5f)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
