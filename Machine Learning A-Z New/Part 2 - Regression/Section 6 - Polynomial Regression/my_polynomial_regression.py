# -*- coding: utf-8 -*-

# Importing libraries
import numpy as np # Mathematics library
import matplotlib.pyplot as plt # Plot charts/graphs
import pandas as pd # Import and manage datasets

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting dataset into test and training set
"""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Feature scaling
"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)"""

# Fitting the linear regression to dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting the polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising linear regression model
plt.scatter(X, y)
plt.plot(X, linear_regressor.predict(X), color = 'black')
plt.title('Truth or bluff? (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show

# Visualising polynomial regression model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y)
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'black')
plt.title('Truth or bluff? (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show

# Predicting a new result with linear regression
linear_regressor.predict([[6.5]])

# Predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
