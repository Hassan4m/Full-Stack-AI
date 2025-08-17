import pandas as pd
import numpy as  np

df = pd.read_csv("E:\\codes\Full-Stack-AI\\csv files\\number-of-registered-medical-and-dental-doctors-by-gender-in-pakistan.csv",
                 index_col="Years")


for col in df.columns:
    df[col] = df[col].astype(str).str.replace(",", "").astype(int)



print("\nDataFrame Info:")
print(df.info())

print("\nData Types:")
print(df.dtypes)

print("\nDescribe:")
print(df.describe())

print("\nShape:")
print(df.shape)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['Female Doctors']].values
y = df[['Female Dentists']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

intercept = model.intercept_[0]
print(f"\nIntercept: {intercept}")

slope = model.coef_[0][0]
print(f"Slope: {slope}")


def predict_female_dentists(female_doctors):
    return slope * female_doctors + intercept

# Test with 3 sample values from dataset
test_values = [7000, 3000, 8000]
for val in test_values:
    predicted = predict_female_dentists(val)
    print(f"Predicted Female Dentists for {val} Female Doctors: {predicted:.2f}")



y_pred = model.predict(X_test)

# Print predicted vs actual
for actual, predicted in zip(y_test.flatten(), y_pred.flatten()):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")




from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nMean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
