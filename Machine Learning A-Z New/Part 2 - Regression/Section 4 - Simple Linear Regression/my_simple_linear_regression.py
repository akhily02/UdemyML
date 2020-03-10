# -*- coding: utf-8 -*-

# Simple linear regression

#DATA PREPROCESSING

# Importing libraries
import numpy as np # Mathematics library
import matplotlib.pyplot as plt # Plot charts/graphs
import pandas as pd # Import and manage datasets

# Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1]

# Splitting dataset into test and training set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Feature scaling
"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)"""

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Visualuze the training set results
plt.scatter(X_train, y_train)
plt.plot(X_train, regressor.predict(X_train), color = 'black')

plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Visualuze the test set results
plt.scatter(X_test, y_test)
plt.plot(X_train, regressor.predict(X_train), color = 'black')

plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

