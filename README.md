# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Import the required libraries and read the dataframe.

 2. Assign hours to X and scores to Y.

 3. Implement training set and test set of the dataframe

 4. Plot the required graph both for test data and training data.

 5. Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:M.PRANATHI 
RegisterNumber:212222240064
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()


df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,regressor.predict(x_test),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![head](https://user-images.githubusercontent.com/118343610/229328806-0a49696a-1bc5-4c8f-ae61-0255e187c85d.png)


![tail](https://user-images.githubusercontent.com/118343610/229328832-25c89c2d-7043-46d6-89bb-15d8a0ae985a.png)


![x](https://user-images.githubusercontent.com/118343610/229328862-27860830-a562-48c4-889b-b84f88011e64.png)


![y](https://user-images.githubusercontent.com/118343610/229328894-a4677e32-a3e3-43eb-a17a-224d8405019b.png)


![ypredct](https://user-images.githubusercontent.com/118343610/229328976-e304ef5c-ea48-4e2f-a214-9ea9cbe4fbc5.png)


![ytest](https://user-images.githubusercontent.com/118343610/229328982-064c7ea0-ea19-4208-8ff1-df7917aac330.png)


![graph1](https://user-images.githubusercontent.com/118343610/229329001-3e169c03-510d-4706-a0e1-d3bacf7bc305.png)


![graph2](https://user-images.githubusercontent.com/118343610/229329017-5b85ceca-bbbd-4155-802a-5595aa230ea2.png)


![mmr](https://user-images.githubusercontent.com/118343610/229329037-31359559-63b5-4046-9c90-643b2467a3eb.png)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
