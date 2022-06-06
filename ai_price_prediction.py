import Crypto
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

###  Get price data  ###
crypto_currency = "BTC"
against_currency = "USD"

start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

data = web.DataReader(f"{crypto_currency}-{against_currency}", "yahoo", start, end)

###  Prepare data for the neural network training  ###
# Scale the data to a 0 to 1 range
scaler = MinMaxScaler(feature_range=(0, 1))
# Use only the close price
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

# Set the amount of days the neural network should be trained on
prediction_days = 60
future_day = 15

x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data) - future_day):
    x_train.append(scaled_data[x - prediction_days : x, 0])
    y_train.append(scaled_data[x + future_day, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

###  Create Neural Network  ###
# pip install numpy==1.19.5
# best for sequential data sets like this
model = Sequential()
LSTM_units = 50
DD_factor = 0.2
Epochs = 100
Batch = 32

# use LongShortTermMemory (LSTM) layers, which are specialized for sequential data
model.add(
    LSTM(units=LSTM_units, return_sequences=True, input_shape=(x_train.shape[1], 1))
)
# Use a dropout layers to reduce the overfitting of the data
model.add(Dropout(DD_factor))
model.add(LSTM(units=LSTM_units, return_sequences=True))
model.add(Dropout(DD_factor))
model.add(LSTM(units=LSTM_units))
model.add(Dropout(DD_factor))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=Epochs, batch_size=Batch)

### Testing the Model  ###
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(
    f"{crypto_currency}-{against_currency}", "yahoo", test_start, test_end
)
actual_prices = test_data["Close"].values

total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

model_inputs = total_dataset[
    len(total_dataset) - len(test_data) - prediction_days :
].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days : x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, color="k", label="Actual Prices")
plt.plot(prediction_prices, color="g", label="Predicted Prices")
plt.title(f"{crypto_currency} price prediction")
plt.xlabel("Days")
plt.ylabel(f"Price / {against_currency}")
plt.legend(loc="upper left")
plt.show()

# Predict for the next day
predict_tomorrow = True
if predict_tomorrow:
    real_data = [
        model_inputs[len(model_inputs) + 1 - prediction_days : len(model_inputs) + 1, 0]
    ]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Predicted price in {future_day}: ", prediction)
