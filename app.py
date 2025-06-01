import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io

st.title("LSTM Temperature Prediction")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Sample Data", df.head())

    if "temperature" not in df.columns:
        st.error("❌ The uploaded CSV must have a 'temperature' column.")
    else:
        run_model = st.button("Run Model")

        if run_model:
            # Smooth
            df['smoothed'] = df['temperature'].rolling(window=3, min_periods=1).mean()

            # Scale
            scaler = MinMaxScaler()
            data = scaler.fit_transform(df['smoothed'].values.reshape(-1, 1))

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

            # Train/test split
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # Model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(seq_length, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            with st.spinner("⏳ Training model..."):
                history = model.fit(X_train, y_train, epochs=200, verbose=0, validation_data=(X_test, y_test))

            # Predict
            y_pred = model.predict(X_test)

            # Inverse scale
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_inv = scaler.inverse_transform(y_pred)

            # Metrics
            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
            st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
            st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")

            # Plot
            fig, ax = plt.subplots()
            ax.plot(y_test_inv, label="Actual")
            ax.plot(y_pred_inv, label="Predicted")
            ax.set_title("Actual vs Predicted Temperatures")
            ax.legend()
            st.pyplot(fig)

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
