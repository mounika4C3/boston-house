# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:44:34 2023

@author: yaman
"""

#REGRESSION ALGORITHM--------> check for error at the end

import numpy as np
import pandas as pd

a=["crim","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]      
     
data=pd.read_csv(r"C:\Users\yaman\Downloads\HousingData.csv",names=a)

s=data.shape
data.size
data.head()
data.info()
data.describe()
a
c=0
combined=[]
for i in range(s[0]):
    l=[]
    for j in range(s[1]):
        l.append(data.iloc[i,j])
    combined.append("".join(map(str,l)))
print(len(combined))
df1=data
df1["COMBINED"]=combined


x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=88)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import mean_squared_error
import math
print(math.sqrt(mean_squared_error(ytest,ypred)))


print(model.predict([[0.49298,0.     ,   9.9    ,   0.     ,   0.544  ,   6.635  ,
        82.5    ,   3.3175 ,   4.     , 304.     ,  18.4    , 396.9    ,
         4.54   ]]))
