# Stock_Market_Prediction_for_NextDay
Stock Price Prediction Project Overview

• Importing libraries like `yfinance`, `tensorflow`, `numpy`, `MinMaxScaler`, and `matplotlib`.
• Data collection: User enters stock symbol, stock data is downloaded and saved to a CSV file.
• Data preprocessing: Stock data is printed and normalized using `MinMaxScaler`.
• Creating sequences: Normalized stock prices are divided into fixed length sequences.
• Train-Test Split: Data is split into training(80%) and testing(20%) sets.
• Building the LSTM Model: Sequential model created using Keras, consisting of an LSTM layer and a Dense layer.
• Compiling and Training the Model: Model compiled with Adam optimizer and MSE loss function. It's then trained on the training data for a specified number of epochs and batch size..
• Making Predictions: Model predicts stock price using the last sequence from test data.
• Plotting Results: Real and predicted prices are inverse-transformed to their original scale and plotted on a graph.
• Displaying Results: The graph displays real and predicted prices, with a title indicating the stock symbol for which the prediction was made.
