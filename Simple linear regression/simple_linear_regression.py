# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:00:19 2017

@author: gavit
"""
# simple linear regression



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values # x is matrix of features i.e independent variable
y = dataset.iloc[:, 1].values

# plotting  graph of x & y values
'''
plt.scatter(x, y, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience ')
plt.xlabel('year of Experience')
plt.ylabel('salary')
plt.show()
 
'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train)
x_test = sc_x.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test set result
y_pred = regressor.predict(x_test)
 

# visualising training set result
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('year of Experience')
plt.ylabel('salary')
plt.show() 


# visualising test set result
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('year of Experience')
plt.ylabel('salary')
plt.show() 

