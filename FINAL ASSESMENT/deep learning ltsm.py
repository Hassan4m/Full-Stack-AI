# Multi Linear Regression Final Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping

#Load and explore the dataset
df = pd.read_csv("E:\codes\Full-Stack-AI\csv files\AirPassengers.csv")
print(df)
print(df.head())    
print(df.info())
print(df.info())
print(df.describe())

# Data Cleaning

# Remove duplicates
df = df.drop_duplicates()

# Remove missing/null values
df = df.dropna(subset=['Month', 'Passengers'])

# Remove outliers (optional, for creative analysis)
Q1 = df['Passengers'].quantile(0.25)
Q3 = df['Passengers'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Passengers'] >= Q1 - 1.5 * IQR) & (df['Passengers'] <= Q3 + 1.5 * IQR)]

print("Cleaned Data:")

#parse date and set as index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Visualize the time series
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x=df.index, y='Passengers')
plt.title('Monthly US Air Passengers (1949-1960)')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.show()

#Feature Engineering: Add Month and Year as features
df['Month_num'] = df.index.month
df['Year'] = df.index.year

# Scaling
scaler = MinMaxScaler()
df['Passengers_scaled'] = scaler.fit_transform(df[['Passengers']])

#Prepare Data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 12  # Use past 12 months to predict next month
data = df['Passengers_scaled'].values
X, y = create_sequences(data, SEQ_LEN)

# Reshape for LSTM [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

#Train-Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#Build LSTM Model
model = Sequential([
    LSTM(64,activation='tanh', input_shape=(SEQ_LEN, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='tanh'),
    Dropout(0.2), 
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[Precision(), Recall()])

#Train Model

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

#Model Metric
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict and inverse scale
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = scaler.inverse_transform(y_pred)

#Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"Test MAE: {mae:.2f}")
print(f"Test MSE: {mse:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R2: {r2:.2f}")

#Plot Actual vs Predicted
plt.figure(figsize=(10,5))
plt.plot(df.index[-len(y_test_inv):], y_test_inv, label='Actual')
plt.plot(df.index[-len(y_test_inv):], y_pred_inv, label='Predicted')
plt.title('Actual vs Predicted US Air Passengers')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.show()

#Forecast Next 12 Months
last_seq = data[-SEQ_LEN:]
future_preds = []
current_seq = last_seq.copy()
for _ in range(12):
    pred = model.predict(current_seq.reshape(1, SEQ_LEN, 1))[0,0]
    future_preds.append(pred)
    current_seq = np.append(current_seq[1:], pred)

future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))
future_dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='MS')

plt.figure(figsize=(10,5))
plt.plot(df.index, df['Passengers'], label='Historical')
plt.plot(future_dates, future_preds_inv, label='Forecast', linestyle='--')
plt.title('Forecasted US Air Passengers for Next 12 Months')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.show()
