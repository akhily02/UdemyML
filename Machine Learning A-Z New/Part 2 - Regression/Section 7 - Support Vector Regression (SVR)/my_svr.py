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
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1)))

# Fitting the svr model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', gamma = 'auto')
regressor.fit(X, y)

# Visualising the svr model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y)
plt.plot(X_grid, regressor.predict(X_grid), color = 'black')
plt.title('Truth or bluff? (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show

# Predicting a new result with the svr model
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
# regressor.predict([[6.5]])
