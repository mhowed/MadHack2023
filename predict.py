'''
Information on TensorFlow, and more specifically LSTM and Dense was found on
https://medium.com/analytics-vidhya/long-short-term-memory-networks-lstm-in-tensorflow-e986dac5cf27
https://www.geeksforgeeks.org/long-short-term-memory-lstm-rnn-in-tensorflow/
https://www.youtube.com/watch?v=IrPhMM_RUmg

'''

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import yfinance as yahoo
import os.path
import absl.logging




def prediction(ticker, useOldModel=False):
    currtime = datetime.now().strftime('%Y-%m-%d')
    df = None
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("images"):
        os.mkdir("images")

    if os.path.isfile(f"data/{ticker}.csv"):
        dateOfFile = datetime.fromtimestamp(os.path.getmtime(
            f"data/{ticker}.csv")).strftime('%Y-%m-%d')

        if dateOfFile < currtime:
            print("Fetching new Data")
            pd.DataFrame(yahoo.download(ticker, "2012-01-01",
                         currtime)).to_csv(f"data/{ticker}.csv")
    else:
        print("Getting new Data")
        pd.DataFrame(yahoo.download(ticker, "2012-01-01", currtime)
                     ).to_csv(f"data/{ticker}.csv")

    df = pd.read_csv(f"data/{ticker}.csv")

    plt.style.use("seaborn-v0_8-bright")

    data = df.filter(["Close"])
    dataset = data.values

    train_split = .80

    train_len = math.ceil(len(dataset) * train_split)

    days = 60

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Checks if a model is saved
    if useOldModel and os.path.isfile(f"data/{ticker}/saved_model.pb"):

        # Check that the model exists
        model = load_model(f"data/{ticker}")

    else:

        train_data = scaled_data[0:train_len, :]

        x_train = []
        y_train = []

        for i in range(days, len(train_data)):
            x_train.append(train_data[i-days:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential([
            LSTM(50, return_sequences =True, input_shape = (x_train.shape[1],1)),
            LSTM(50, return_sequences = False),
            Dense(25),
            Dense(1)
        ])

        # compile
        model.compile(optimizer='adamax', loss='mse')

        # fit using x,y,every piece of data,only 1 run
        model.fit(x_train, y_train, batch_size=1, epochs=1)


    test = scaled_data[train_len - days:, :]

    # Create the data set x_test and y_test
    x_test = []
    y_test = dataset[train_len:, :]
    for i in range(days, len(test)):
        x_test.append(test[i-days:i, 0])

    # make it a numpy array
    x_test = np.array(x_test)

    # reshape data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    rmse
    print("RMSE:",rmse)

    # get data we didnt touch
    valid = data[train_len:]
    # data that was predicted
    train = data[:train_len]
    valid['Predictions'] = predictions

    #print(model.summary())
    return train, valid, predictions, ticker, model


def plot(train, valid, predictions, ticker):
    # plot data
    plt.figure(figsize=(10, 6))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Real', 'Predictions'], loc='upper left')
    plt.savefig(f"images/{ticker}.png")

    print("Prediction for next close:", str(predictions[-1])[1:-1])


def main():
    pd.options.mode.chained_assignment = None
    absl.logging.set_verbosity(absl.logging.ERROR)


    train, valid, predictions, ticker, model = prediction('AAPL', True)

    # make sure good start
    x = valid.head(1)
    real = x["Close"].values
    pred = x["Predictions"].values
    print("Real / Pred:", real/pred)

    while real/pred > 1.02 or real/pred < 0.98:
        train, valid, predictions, ticker, model= prediction('AAPL')
        x = valid.head(1)
        real = x["Close"].values
        pred = x["Predictions"].values
        print("Real / Pred:", real/pred)

    model.save(f"data/{ticker}")

    plot(train, valid, predictions, ticker)


if __name__ == "__main__":
    main()
