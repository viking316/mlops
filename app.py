import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),'src'))

from logger import logger  # Import your configured logger

st.title("LSTM Temperature Prediction")
logger.info("Streamlit app started")

uploaded_file = st.file_uploader("Upload your CSV")

if uploaded_file is not None:
    logger.info(f"File uploaded: {uploaded_file.name}")
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Sample Data", df.head())
        logger.debug(f"CSV head:\n{df.head()}")

        if "temperature" not in df.columns:
            st.error("❌ The uploaded CSV must have a 'temperature' column.")
            logger.warning("CSV missing required 'temperature' column")
        else:
            run_model = st.button("Run Model")
            if run_model:
                logger.info("Run Model button clicked")

                try:
                    # Smooth
                    df['smoothed'] = df['temperature'].rolling(window=3, min_periods=1).mean()
                    logger.debug("Smoothing applied")

                    # Scale
                    scaler = MinMaxScaler()
                    data = scaler.fit_transform(df['smoothed'].values.reshape(-1, 1))
                    logger.debug("Data normalized")

                    # Sequence creation
                    def create_sequences(data, seq_length):
                        X, y = [], []
                        for i in range(len(data) - seq_length):
                            X.append(data[i:i+seq_length])
                            y.append(data[i+seq_length])
                        return np.array(X), np.array(y)

                    seq_length = 3
                    X, y = create_sequences(data, seq_length)
                    X = X.reshape((X.shape[0], X.shape[1], 1))
                    logger.info(f"Created sequences: X shape = {X.shape}, y shape = {y.shape}")

                    # Train/test split
                    split_index = int(len(X) * 0.8)
                    X_train, X_test = X[:split_index], X[split_index:]
                    y_train, y_test = y[:split_index], y[split_index:]
                    logger.debug("Data split into training and testing")

                    # Model
                    model = Sequential([
                        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    logger.info("Model compiled")

                    with st.spinner("⏳ Training model..."):
                        history = model.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_test, y_test))
                    logger.info("Model training completed")

                    # Predict
                    y_pred = model.predict(X_test)
                    logger.info("Prediction completed")

                    # Inverse scale
                    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
                    y_pred_inv = scaler.inverse_transform(y_pred)

                    # Metrics
                    mae = mean_absolute_error(y_test_inv, y_pred_inv)
                    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
                    st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
                    st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
                    logger.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

                    # Plot
                    fig, ax = plt.subplots()
                    ax.plot(y_test_inv, label="Actual")
                    ax.plot(y_pred_inv, label="Predicted")
                    ax.set_title("Actual vs Predicted Temperatures")
                    ax.legend()
                    st.pyplot(fig)
                    logger.debug("Plot rendered")

                    # Download CSV
                    results_df = pd.DataFrame({
                        "Actual": y_test_inv.flatten(),
                        "Predicted": y_pred_inv.flatten()
                    })

                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()

                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv_data,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    logger.info("Download button displayed")

                except Exception as e:
                    st.error("An error occurred while processing the model.")
                    logger.error(f"Model processing error: {e}")

    except Exception as e:
        st.error("Failed to load CSV.")
        logger.error(f"CSV loading error: {e}")
else:
    logger.debug("No file uploaded yet")
#hello test
