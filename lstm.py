import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load CSV
df = pd.read_csv("Offset equipment temperature degrees Celsius observed every 10 minutes.csv")

# Smooth the temperature data using a rolling mean
df['smoothed'] = df['temperature'].rolling(window=3, min_periods=1).mean()

# Normalize the smoothed data
scaler = MinMaxScaler()
data = scaler.fit_transform(df['smoothed'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 3
X, y = create_sequences(data, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into training and testing
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=200, verbose=0, validation_data=(X_test, y_test))

# Predict on test set
y_pred = model.predict(X_test)

# Inverse transform for comparison
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Print change in loss
loss_start = history.history['loss'][0]
loss_end = history.history['loss'][-1]
print(f"Loss change: {loss_start:.6f} -> {loss_end:.6f} (Δ = {loss_start - loss_end:.6f})\n")

# Show predictions vs actual
for i in range(len(y_test_inv)):
    print(f"Expected: {y_test_inv[i][0]:.2f}, Predicted: {y_pred_inv[i][0]:.2f}")

# Optional: Plot
plt.plot(y_test_inv, label="Actual")
plt.plot(y_pred_inv, label="Predicted")
plt.title("Actual vs Predicted Temperatures")
plt.legend()
plt.show()

