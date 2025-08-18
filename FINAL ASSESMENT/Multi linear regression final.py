
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

# --- Load the dataset ---
# Reading the house sales data from a CSV file
df = pd.read_csv(r"E:\codes\Full-Stack-AI\csv files\us_house_Sales_data.csv", delimiter=',')

# --- Data Cleaning ---
# Remove dollar sign from 'Price' and convert to float
df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)
# Remove text from 'Bedrooms', 'Bathrooms', 'Area (Sqft)', and 'Lot Size' and convert to numeric
df['Bedrooms'] = df['Bedrooms'].str.replace(' bds?', '', regex=True).astype(int)
df['Bathrooms'] = df['Bathrooms'].str.replace(' ba', '', regex=True).astype(float)
df['Area (Sqft)'] = df['Area (Sqft)'].str.replace(' sqft', '', regex=True).astype(float)
df['Lot Size'] = df['Lot Size'].str.replace(' sqft', '', regex=True).astype(float)

# --- Data Overview ---
# Print the cleaned dataset and some basic info
print("Cleaned Data Preview:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nShape:", df.shape)
print("\nData Types:\n", df.dtypes)

# --- Feature Selection ---
# Selecting the features (independent variables) and the target (dependent variable)
X = df[['Bedrooms', 'Bathrooms', 'Area (Sqft)', 'Lot Size', 'Year Built', 'Days on Market']]
y = df['Price']

#Train Test Split 
# Splitting the data into training and testing sets (80%,20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Training
# Creating and training the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Prediction & Evaluation
# Predicting house prices on the test set
y_pred = model.predict(X_test)
# Printing evaluation metrics for the model
print("\nModel Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

#Model Coefficients
# Displaying the learned coefficients for each feature
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nModel Coefficients:")
print(coefficients)
print("Intercept:", model.intercept_)

#Visualization: Actual vs Predicted Prices
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
# Plotting a diagonal line for perfect prediction
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

#Visualization: Residuals Plot 
# Plotting the residuals to check for patterns
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

#Visualization: Bedrooms vs Price
# Exploring the relationship between number of bedrooms and price
plt.figure(figsize=(8,6))
plt.scatter(df['Bedrooms'], df['Price'], alpha=0.7)
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.title('Bedrooms vs Price')
plt.show()
