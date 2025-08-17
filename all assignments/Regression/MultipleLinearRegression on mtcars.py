import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('E:\codes\Full-Stack-AI\csv files\mtcars.csv')

# Step 12: DataFrame inspection
print("\n--- .info() ---")
print(df.info())
print("\n--- .dtypes ---")
print(df.dtypes)
print("\n--- .describe() ---")
print(df.describe())
print("\n--- .shape ---")
print(df.shape)

# Step 13: Assumption (e.g., 'mpg' is the dependent, 'hp', 'wt', 'disp' are independent)
X = df[['hp', 'wt', 'disp']].values
y = df['mpg'].values

# Step 14: Regression plot
sns.pairplot(df, x_vars=['hp', 'wt', 'disp'], y_vars='mpg', kind='reg')
plt.suptitle('Regression Plots for Independent Variables vs. MPG', y=1.02)
plt.show()

# Step 15: Correlation heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(df.drop(columns=['model']).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# Step 16: Split data (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# Step 17: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 18: Intercept
print("\nIntercept:", model.intercept_)

# Step 19: Slope (coefficients)
print("Slope (coefficients):", model.coef_)

# Step 20: Predict for test data
y_pred = model.predict(X_test)
print("\nPredicted values:", y_pred)
print("Actual values:   ", y_test)

# Step 21: Metrics (MAE, MSE, RMSE)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n--- Performance Metrics ---")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

print("\n actyal vs predicted")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()
