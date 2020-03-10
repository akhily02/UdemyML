# -*- coding: utf-8 -*-

# Importing libraries
import numpy as np # Mathematics library
import matplotlib.pyplot as plt # Plot charts/graphs
import pandas as pd # Import and manage datasets

# Importing the dataset
dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4]

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# Dummy encoding
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting dataset into test and training set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature scaling
"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)"""

# Fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Building optimal model using backward elimination
import statsmodels.formula.api as sm
# For the constant, since not taken care in statsmodel
X = np.append(values = X, arr = np.ones((50,1)).astype(int), axis = 1)

# considering significance level of 5% (0.05)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

# Remove index 2 which has highest P value which is greater than significance level
X_opt = X[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

# Remove index 1 which has highest P value which is greater than significance level
X_opt = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

# Remove index 4 which has highest P value which is greater than significance level
X_opt = X[:,[0,3,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

# Remove index 5 which has highest P value which is slightly greater than significance level
X_opt = X[:,[0,3]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()
