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

# Fitting the regression model to dataset


# Visualising the regression model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y)
plt.plot(X_grid, regressor.predict(X_grid), color = 'black')
plt.title('Truth or bluff? (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show

# Predicting a new result with the regression model
regressor.predict([[6.5]])
