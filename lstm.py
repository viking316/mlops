import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Original dataset
data = [
    22, 22.8, 22.9, 22.5, 22.5, 22, 21.9, 21.9, 22.3, 22.5,
    22.4, 21.9, 21.5, 21.5, 21.3, 21, 21, 21, 21, 20.5
]

#data loaded
# Normalize data
data = np.array(data)
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Convert to supervised learning format
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 3
X, y = create_sequences(data, seq_length)

# Reshape input to [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=200, verbose=0)

# Predict next value
last_sequence = data[-seq_length:].reshape((1, seq_length, 1))
predicted = model.predict(last_sequence, verbose=0)
predicted_value = predicted[0][0] * (np.max(data) - np.min(data)) + np.min(data)

print(f"Predicted next value: {predicted_value:.2f}")

# Optional: Plot original vs predicted sequence
plt.plot(range(len(data)), data, label="Original")
plt.axvline(len(data)-1, color='r', linestyle='--')
plt.plot(len(data), predicted[0][0], 'go', label="Prediction")
plt.legend()
plt.title("LSTM Prediction")
plt.show()
