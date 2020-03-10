# -*- coding: utf-8 -*-

# Importing libraries
import numpy as np # Mathematics library
import matplotlib.pyplot as plt # Plot charts/graphs
import pandas as pd # Import and manage datasets

# Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)
