import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv('E:\\codes\Full-Stack-AI\\csv files\\50_Startups.csv')
print(df)

print("\n--- .info() ---")
print(df.info())

print("\n--- .dtypes ---")
print(df.dtypes)

print("\n--- .describe() ---")
print(df.describe())

print("\n--- .shape ---")
print(df.shape)

# 13. Independent and Dependent Variables
# Drop 'State' column (categorical)

df = df.drop("State", axis=1)


# Convert to array format for sklearn
X = df.drop("Profit", axis=1).values
y = df["Profit"].values

# 14. Best-fitting regression line (using only R&D Spend for regplot)
sns.regplot(x=df["R&D Spend"], y=df["Profit"])
plt.title("Best-Fitting Line: R&D Spend vs Profit")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.show()

#Correlation heatmap
plt.figure(figsize = (8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()



# 16. Split data
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 17. Train Linear Regression model
model = LinearRegression()
model.fit(X_train,y_train)

# 18. Print Intercept
print("\n--- Intercept ---")
print(model.intercept_)

# 19. Print Coefficients
print("\n--- Coefficients ---")
for features,coef in zip(["R&D Spend","Administration","Marketing Spend"],model.coef_):
	print(f"{features}: {coef}")

# 20. Predict Profit
y_pred = model.predict(X_test)
print("\n--- Predicted Profit ---")
for actual,predicted in zip(y_test,y_pred):
    print(f"Actual: {round(actual,2)}, Predicted: {round(predicted,2)}")

# 21. Metric Analysis
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)

print("\n--- Metrics ---")
print(f"MAE: {round(mae,2)}")
print(f"MSE: {round(mse,2)}")
print(f"RMSE: {round(rmse,2)}")

# 22. Visualize Actual vs Predicted Profits

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Profits')
plt.xlabel('Actual Profits')
plt.ylabel('Predicted Profits')
plt.grid()
plt.show()