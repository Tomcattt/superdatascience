# Upper Confidence Bound

# Importing the librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/quentin/Documents/AI/UDEMY_ML/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv')

N,d = dataset.shape

# Implementing UCB
numbers_of_selection = np.array([0] * d)
sums_of_rewards = np.array([0] * d)
ads_selected = []
total_reward = 0
for n in range(0,N):
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        if numbers_of_selection[i] > 0:
            average_reward = sums_of_rewards[i]/numbers_of_selection[i]
            delta_i = np.sqrt(3/2 * np.log(n+1)/numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound :
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selection[ad] += 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
    


