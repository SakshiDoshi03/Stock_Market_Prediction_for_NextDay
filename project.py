import yfinance as yf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

print("TO DOWNLOAD INDIAN STOCKS WRITE .NS AFTER STOCK SYMBOL")
print("***************************************")

stock_symbol = input("Enter the stock symbol: ")

# DOWNLOAD STOCK
stock_data = yf.download(stock_symbol, period="max")
stock_data.reset_index(inplace=True)
stock_data.to_csv(stock_symbol + ".csv")

# PRINT DATA OF STOCK
print(stock_data)
# PRINT LENGTH
length = len(stock_data)
print("SEQUENCE OF DATA", length)

close_prices = stock_data['Close'].values.reshape(-1, 1)

# NORMALIZING DATA USING MINMAXSCALER
# MINMAXSCALER SCALES DATA BETWEEN FIX RANGE - 0 OR 1
scaler = MinMaxScaler()
close_prices_scaled = scaler.fit_transform(close_prices)

sequence_length = 10
X = []
y = []
for i in range(len(close_prices_scaled) - sequence_length):
    X.append(close_prices_scaled[i:i+sequence_length])
    y.append(close_prices_scaled[i+sequence_length])

X = np.array(X)
y = np.array(y)

# SPLIT MODEL INTO TRAINING AND TESTING SET
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = keras.Sequential([
    keras.layers.LSTM(50, activation="relu", input_shape=(sequence_length, 1)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=45, batch_size=30, verbose=1)

# TO MAKE PREDICTION FOR LAST DAY WE'LL TAKE LAST SEQUENCE FROM X_TEST
last_sequence = X_test[-1]
last_sequence = last_sequence.reshape(1, sequence_length, 1)

# PRINT NEXT DAY PREDICTION
next_day_prediction = model.predict(last_sequence)


next_day_price = scaler.inverse_transform(next_day_prediction)
print("Predicted Price for Next Day:", next_day_price)

predicted_prices = model.predict(X_test)

# TO PLOT GRAPH WE HAVE TO TRANSFORM VALUES INTO ORIGINAL VALUES
predicted_prices = scaler.inverse_transform(predicted_prices)
real_close_prices = scaler.inverse_transform(y_test)

plt.plot(real_close_prices, color="blue", label="Close price")
plt.plot(predicted_prices, color="red", label="Predicted price")
plt.title("Stock price prediction for "+stock_symbol)
plt.xlabel("DURATION")
plt.ylabel("PRICE")
plt.legend()
plt.show()
