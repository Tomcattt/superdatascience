# Thomson Sampling

# Importing the librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('/Users/quentin/Documents/AI/UDEMY_ML/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv')

N,d = dataset.shape

# Implementing UCB
number_of_rewards_1 = np.array([0] * d)
number_of_rewards_0 = np.array([0] * d)

ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = 0
    max_random = 0
    for i in range(0,d):
        random_beta = random.betavariate(1+number_of_rewards_1[i], 1+number_of_rewards_0[i])
        if random_beta > max_random :
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward == 1:
        number_of_rewards_1[ad] += 1
    else :
        number_of_rewards_0[ad] += 1
    total_reward += reward
    

# Visualizing the results

plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
