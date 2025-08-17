import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("E:\\codes\Full-Stack-AI\\all assignments\\student_scores.csv")
print(df.head())  # print first five rows of dataframe
print("shape of the data: ", df.shape)  # print shape of the data rows, columns

df.plot.scatter(x="Hours", y="Scores")  # plot a scatter graph of the data
plt.title("Scatter Plot")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()  # show the plot

# understanding correlation between the two variables
# Correlation tells us how strongly two variables are related.

# Values range from:

# +1: strong positive

# -1: strong negative

# 0: no relationship
print("correlation between hours and scores:")
print(df.corr())  # print correlation between hours and scores

print("\n describe the data:", df.describe())


X = df["Hours"].values.reshape(-1, 1)  # reshape the data to be a 2D array
y = df["Scores"].values.reshape(-1, 1)  # reshape the data to be a 2D array


print("2d x array " ,X)
print("2d y array " ,y)

print("value of hours",df["Hours"].values)  # print the values of hours
print(" shape of hours",df["Hours"].values.shape)  


print("shape of X",X.shape)
print("X", X)

SEED = 42  # set seed for reproducibility
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y , test_size=0.2, random_state=SEED) # split the data into training and testing sets
print("X_train data",X_train)
print("y_train data",y_train)

from sklearn.linear_model import LinearRegression
