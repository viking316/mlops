# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, window=3):
    df['smoothed'] = df['temperature'].rolling(window=window, min_periods=1).mean()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df['smoothed'].values.reshape(-1, 1))
    return data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return X, y
