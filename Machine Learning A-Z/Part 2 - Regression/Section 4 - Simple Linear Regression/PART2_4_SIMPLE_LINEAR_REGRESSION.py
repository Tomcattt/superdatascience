#Data Preprocessing

#importing the librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3,random_state = 0)

# feature scaling
"""from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)"""

# Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train,y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of experiences')
plt.ylabel('Salary $USD')
plt.show()

# Visualising the Test set results
plt.scatter(X_test,y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Years of experiences')
plt.ylabel('Salary $USD')
plt.show()