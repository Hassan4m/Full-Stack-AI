import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('E:\codes\Full-Stack-AI\csv files\Real_Estate_Sales_2001-2022_GL-Short (2).csv' ,
                  index_col="Serial Number"
                  )


print(df.info())         # Info about nulls, types
print(df.dtypes)         # Data types
print(df.describe())     # Summary stats
print(df.shape)          # Shape (rows, columns)


# Features (X) and Labels (Y)
X = df[['Assessed Value']].values  # 2D array
y = df['Sale Amount'].values       # 1D array



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # Check the shapes of the splits


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

intercept = model.intercept_
print(f"Intercept: {intercept}")

slope = model.coef_[0]
print(f"Slope: {slope}")

# Prediction function
def predict_sale_amount(assessed_value):
    return intercept + slope * assessed_value

# Prediction for 3 sample values
sample_values = [54000, 124000, 203000]
for val in sample_values:
    print(f"Predicted Sale Amount for Assessed Value {val} = {predict_sale_amount(val)}")



y_pred = model.predict(X_test)

# Show predictions
print("Actual vs Predicted Sale Amounts:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")


plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

r2 = model.score(X_test, y_test)
print(f"R² Score: {r2}")
