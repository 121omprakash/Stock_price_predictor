import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

st.title("Stock Price Predictor App")

st.write("""
### About Indian Stock Market
The National Stock Exchange (NSE) is India's leading stock exchange. It was established in 1992 and became fully operational in 1994. 
The NSE's flagship index is the NIFTY 50, which includes 50 of the largest Indian companies across various sectors.

### Key Points:
- Trading Hours: Monday to Friday, 9:15 AM to 3:30 PM IST
- Currency: Indian Rupee (INR)
- Regulator: Securities and Exchange Board of India (SEBI)

Remember to conduct thorough research and consider seeking advice from financial experts before making any investment decisions.
""")

stock = st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

google_data = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data.tail())

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)


def predict_next_10_days(model, last_sequence, scaler):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(10):
        # Reshape the sequence for the model
        current_sequence_reshaped = np.reshape(current_sequence, (1, current_sequence.shape[0], 1))
        
        # Get the prediction
        prediction = model.predict(current_sequence_reshaped)
        
        # Add the prediction to our list
        predictions.append(prediction[0][0])
        
        # Update the sequence
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = prediction
    
    # Inverse transform the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions.flatten()

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)


# Add this after your existing Streamlit code, just before the end of the script

st.subheader("Last 10 Days - Original vs Predicted")

# Get the last 10 days of data
last_10_days = ploting_data.tail(10)

# Display the data
st.write(last_10_days)

# Create a plot for the last 10 days
fig_last_10, ax = plt.subplots(figsize=(12, 6))
ax.plot(last_10_days['original_test_data'], label='Original')
ax.plot(last_10_days['predictions'], label='Predicted')
ax.set_title('Last 10 Days - Original vs Predicted Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig_last_10)


st.subheader("Predictions for Next 10 Days")

# Get the last 100 days of scaled data
last_sequence = scaled_data[-100:]

# Predict next 10 days
next_10_days_predictions = predict_next_10_days(model, last_sequence, scaler)

# Create a date range for the next 10 days
last_date = google_data.index[-1]
date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10)

# Create a DataFrame with the predictions
next_10_days_df = pd.DataFrame({
    'Date': date_range,
    'Predicted Close': next_10_days_predictions
})

# Display the predictions
st.write(next_10_days_df)

# Plot the predictions
fig_next_10, ax = plt.subplots(figsize=(12, 6))
ax.plot(next_10_days_df['Date'], next_10_days_df['Predicted Close'], label='Predicted')
ax.set_title('Predicted Close Price for Next 10 Days')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig_next_10)

