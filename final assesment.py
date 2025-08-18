
# Final Assessment - Clean and Robust Version
""""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv(r"E:\codes\Full-Stack-AI\csv files\us_house_Sales_data.csv", delimiter=',')
df.columns = df.columns.str.strip()

# Show columns and head for debugging
print("Columns in DataFrame:", df.columns.tolist())
print(df.head())

# Data Cleaning
df.drop_duplicates(inplace=True)

# Convert 'Price' to numeric and remove '$' sign
if 'Price' in df.columns:
	df['Price'] = df['Price'].replace('[\$,]', '', regex=True)
	df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Clean 'City' and 'State' columns if they exist
for col in ['City', 'State']:
	if col in df.columns:
		df[col] = df[col].astype(str).str.strip()

# Drop rows with missing values in any column used for regression
df.dropna(subset=['Price'], inplace=True)
df.dropna(axis=0, inplace=True)

# Encode categorical columns (except 'Price')
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [col for col in cat_cols if col != 'Price']
if cat_cols:
	df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Data visualization: House Price Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Price'], bins=30, color='skyblue', edgecolor='black')
plt.title('House Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Multiple Linear Regression
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.4f}')

# Visualizing Predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Saving the model
joblib.dump(model, 'house_price_model.pkl')

# Load the model and make predictions
loaded_model = joblib.load('house_price_model.pkl')
sample_data = X_test.iloc[:5]
predictions = loaded_model.predict(sample_data)
print("Sample Predictions:", predictions)

# Saving the predictions to a CSV file
predictions_df = pd.DataFrame({'Actual': y_test.iloc[:5].values, 'Predicted': predictions})
predictions_df.to_csv('house_price_predictions.csv', index=False)
print(predictions_df)
conclusion = "The model has been successfully trained and evaluated. Predictions have been saved to 'house_price_predictions.csv'."
print(conclusion)   
# """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('"E:\codes\Full-Stack-AI\csv files\us_house_Sales_data.csv"', delimiter=',')


# Clean numeric columns
df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)
df['Bedrooms'] = df['Bedrooms'].str.replace(' bds?', '', regex=True).astype(int)
df['Bathrooms'] = df['Bathrooms'].str.replace(' ba', '', regex=True).astype(float)
df['Area (Sqft)'] = df['Area (Sqft)'].str.replace(' sqft', '', regex=True).astype(float)
df['Lot Size'] = df['Lot Size'].str.replace(' sqft', '', regex=True).astype(float)

# Display the cleaned dataset
print(df)

# Display the first few rows of the dataset
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
print(df.dtypes)

# select independent and dependent variables 
X = df[['Bedrooms', 'Bathrooms', 'Area (Sqft)', 'Lot Size', 'Year Built', 'Days on Market']]
y = df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train multi linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))


# Evaluate the model
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients)
print("Intercept:", model.intercept_)

#  actual vs predicted prices
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.show()

# residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

#  feature vs price
plt.figure(figsize=(8,6))
plt.scatter(df['Bedrooms'], df['Price'], alpha=0.7)
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.title('Bedrooms vs Price')
plt.show()

'''conclusion
In this analysis, we applied multiple linear regression to predict house prices
usingfeatures such as bedrooms, bathrooms, and square footage.
The model demonstrated how these variables influence property prices.
Visualization of actual vs. predicted prices and residuals indicated 
the models effectiveness and highlighted areas for improvement.
Overall, multiple linear regression provides valuable insights
for understanding and forecasting real estate prices, 
though further refinement and inclusion of additional features could enhance prediction accuracy.'''
