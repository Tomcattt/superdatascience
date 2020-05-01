# Apriori Algorithm

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('/Users/quentin/Documents/AI/UDEMY_ML/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori/')


# Importing the dataset
dataset = pd.read_csv('/Users/quentin/Documents/AI/UDEMY_ML/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)