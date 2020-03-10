# -*- coding: utf-8 -*-

# Importing libraries

# import numpy as np # Mathematics library
# import matplotlib.pyplot as plt # Plot charts/graphs
import pandas as pd # Import and manage datasets

# Importing the dataset
dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,3]


# missing data

# sklearn has lot of libraries for creating ml models, 
# preprocessing library contains classes, models to preprocess datasets
# imputer for missing data
from sklearn.preprocessing import Imputer 

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)

# Encoding categorical variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Dummy encoding
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting dataset into test and training set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

