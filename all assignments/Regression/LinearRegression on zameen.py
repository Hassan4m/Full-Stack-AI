import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

#Load the CSV 
df = pd.read_csv("E:\codes\Full-Stack-AI\csv files\zameen_data.csv",
                  index_col="property_id" ,
                    delimiter=';')
print(df)

#  Analyze DataFrame
print("\nData Info:\n", df.info())
print("\nData Types:\n", df.dtypes)
print("\nDescribe:\n", df.describe())
print("\nShape:\n", df.shape)

#Prepare data for Linear Regression
df = df[['bedrooms', 'price']].dropna()
X = df[['bedrooms']].values  # Independent variable
y = df['price'].values       # Dependent variable

#Split into train/test 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Print intercept
intercept = model.intercept_
print("\nIntercept (b):", intercept)

#Print slope
slope = model.coef_[0]
print("Slope (m):", slope)

# Predict price function
def predict_price(bedrooms, slope, intercept):
    return slope * bedrooms + intercept

# Predicting price for 3 values
for b in [2, 4, 6]:
    price = predict_price(b, slope, intercept)
    print(f"Predicted price for {b} bedrooms: {price}")

#Predict for test data
y_pred = model.predict(X_test)

#Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print("\nMAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)