import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('E:\codes\Full-Stack-AI\csv files\housing.csv')

# Step 12: Data inspection
print(df.info())
print(df.dtypes)
print(df.describe())
print(df.shape)

# Step 13: Feature Selection and Imputation
independent_vars = ['median_income', 'total_rooms', 'housing_median_age']
dependent_var = 'median_house_value'

X = df[independent_vars]
y = df[dependent_var]

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
y_array = y.values

# Step 14: Regression plots for each feature
plt.figure(figsize=(18, 5))
for i, var in enumerate(independent_vars):
    plt.subplot(1, 3, i+1)
    sns.regplot(x=df[var], y=y, line_kws={"color": "red"})
    plt.title(f'{var} vs {dependent_var}')
plt.tight_layout()
plt.show()

# Step 15: Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df[independent_vars + [dependent_var]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
 
# Step 16: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_array, test_size=0.1, random_state=42)

# Step 17: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 18: Intercept
print("Intercept:", model.intercept_)

# Step 19: Slope / Coefficients
print("Coefficients (slope):", model.coef_)
for name, coef in zip(independent_vars, model.coef_):
    print(f"{name}: {coef}")

# prediction
y_pred = model.predict(X_test)

#Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

print("\n ---actual vs predicted---")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {round(actual)}, Predicted: {round(predicted)}")
#Residuals Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y_test - y_pred, color='blue', s=10)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()
