# Stock_Market_Prediction_for_NextDay
This project aims to predict stock prices using a Long Short-Term Memory (LSTM) neural network model. Here's a breakdown of the steps involved:

**Importing Libraries**: The required libraries like `yfinance` for fetching stock data, `tensorflow` for building the neural network model, `numpy` for numerical operations, `MinMaxScaler` for normalization, and `matplotlib` for visualization are imported.
**Data Collection**: The user is prompted to enter a stock symbol, and the corresponding stock data is downloaded using Yahoo Finance (`yfinance`). The data is then saved to a CSV file for future use.
**Data Preprocessing**: The stock data is printed and normalized using `MinMaxScaler` to scale the data between 0 and 1, which is a common preprocessing step for neural networks.
**Creating Sequences**: The normalized stock prices are divided into sequences of a fixed length (`sequence_length`). Input sequences (`X`) and corresponding output sequences (`y`) are created by shifting the data with a sliding window approach.
**Train-Test Split**: The data is split into training and testing sets. The training set comprises 80% of the data, and the testing set comprises the remaining 20%.
**Building the LSTM Model**: A sequential model is created using Keras, consisting of an LSTM layer with 50 units followed by a Dense layer.
**Compiling and Training the Model**: The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss function. It's then trained on the training data for a specified number of epochs and batch size.
**Making Predictions**: The model predicts the stock price for the next day using the last sequence from the test data. The prediction is then inverse-transformed to get the actual price.
**Plotting Results**: Both the real stock prices and the predicted prices are inverse-transformed to their original scale. They are then plotted on a graph using Matplotlib to visualize the performance of the model.
**Displaying Results**: The graph displays the real stock prices in blue and the predicted prices in red, with a title indicating the stock symbol for which the prediction was made.

This project demonstrates a basic implementation of using LSTM neural networks for stock price prediction, including data preprocessing, model building, training, and evaluation. 
