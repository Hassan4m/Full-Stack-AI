# Multi Linear Regression Final Model
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf 
from tensorflow import keras
# from tensorflow.keras import sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
from keras.metrics import Precision ,Recall
#dataset
df = pd.read_csv('E:\codes\Full-Stack-AI\csv files\AirPassengers.csv')
print(df.head())

# Design Model with related layers,
#  Performing core operation of Tensflow-Keras based deep learning, Perform Model metric analysis
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df['Passengers'] = df['Passengers'].astype(float)
# Splitting the dataset into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]
# Reshape the data for LSTM
X_train = train['Passengers'].values.reshape(-1, 1)
X_test = test['Passengers'].values.reshape(-1, 1)
# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Reshape the data for LSTM input
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
# Define the LSTM model
model = keras.Sequential([
    keras.layers.LSTM(50, activation='relu', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[Precision(), Recall()])
# Train the model
history = model.fit(X_train_scaled, train['Passengers'].values, epochs=50, batch_size=32, validation_data=(X_test_scaled, test['Passengers'].values), verbose=1)
# Evaluate the model
loss, precision, recall = model.evaluate(X_test_scaled, test['Passengers'].values, verbose=0)
print(f"Test Loss: {loss}, Precision: {precision}, Recall: {recall}")
# Make predictions
predictions = model.predict(X_test_scaled)
# Inverse transform the predictions
predictions = scaler.inverse_transform(predictions)
# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index[:train_size], train['Passengers'], label='Train Data')
plt.plot(df.index[train_size:], test['Passengers'], label='Test Data')
plt.plot(df.index[train_size:], predictions, label='Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('Air Passengers Prediction using LSTM')
plt.legend()
plt.show()
# Save the model
model.save('air_passengers_lstm_model.h5')
# Load the model
loaded_model = keras.models.load_model('air_passengers_lstm_model.h5')
# Make predictions with the loaded model
loaded_predictions = loaded_model.predict(X_test_scaled)
# Inverse transform the loaded predictions
loaded_predictions = scaler.inverse_transform(loaded_predictions)
# Plot the results with the loaded model
plt.figure(figsize=(12, 6))
plt.plot(df.index[:train_size], train['Passengers'], label='Train Data')
plt.plot(df.index[train_size:], test['Passengers'], label='Test Data')  
plt.plot(df.index[train_size:], loaded_predictions, label='Loaded Model Predictions', color='green')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('Air Passengers Prediction using Loaded LSTM Model')
plt.legend()
plt.show()
# Save the predictions to a CSV file
predictions_df = pd.DataFrame({'Date': df.index[train_size:], 'Predicted_Passengers': predictions.flatten()})
predictions_df.to_csv('air_passengers_predictions.csv', index=False)
print("Predictions saved to 'air_passengers_predictions.csv'")
# This code is a complete example of a deep learning model using LSTM to predict air passenger numbers.