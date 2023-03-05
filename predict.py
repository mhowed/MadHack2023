'''
Information on TensorFlow, and more specifically LSTM and Dense was found on
https://medium.com/analytics-vidhya/long-short-term-memory-networks-lstm-in-tensorflow-e986dac5cf27
https://www.geeksforgeeks.org/long-short-term-memory-lstm-rnn-in-tensorflow/
https://www.youtube.com/watch?v=IrPhMM_RUmg
https://datascience.stackexchange.com/questions/46124/what-do-compile-fit-and-predict-do-in-keras-sequential-models

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



""" Prediction Function used to predict the stock price of the given stock

This function uses the Keras model from Tensorflow with the Adams optimizer. 
It first checks to see if the data has been downloaded and then updates it to be
the latest if the data is over a day old. It then checks to see if you
have a previous Keras Model saved, and if so, if you want to use ir or not.
It then runs the Model using a 60 day training period with a train split of 80%
"""

def prediction(ticker, useOldModel=False):

    # gets the current date
    currtime = datetime.now().strftime('%Y-%m-%d')
    df = None
    # Creates a data directory if the user doesn't have one already
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("images"):
        os.mkdir("images")

    # checks if the data was downloaded, if not, it downloads it
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

    # Reads the data as a pandas DataFrame
    df = pd.read_csv(f"data/{ticker}.csv")

    # Sets the style for the plot to use
    plt.style.use("seaborn-v0_8-bright")

    # gets the Closing stock prices of the data and converts into back to an array
    data = df.filter(["Close"])
    dataset = data.values

    # Sets the train split to train with 80% of the available data
    train_split = .80

    train_len = math.ceil(len(dataset) * train_split)

    days = 60

    # Scales the data from 0,1 using MinMaxScaler() to reduce run time as we are using a gradient optimizer
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(dataset, dataset.shape)

    # Checks if a model is saved and the user want's to use the previous model
    if useOldModel and os.path.isfile(f"data/{ticker}/saved_model.pb"):
        model = load_model(f"data/{ticker}")

    else:
        # Create a new model by first getting a train_data from the scaled_data with only the percent allocated for training
        train_data = scaled_data[0:train_len, :]

        x_train = []
        y_train = []

        # Creates the x_train and y_train data sets by making x_train into a array of arrays that hold 60 scaled closing prices.
        # This is then compared to the 61st day, which is stored in the y_train at the same index.
        for i in range(days, len(train_data)):
            x_train.append(train_data[i-days:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape that is apparently neccessary for the model to work correctly using LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Use a 4 layered Neural Network to create the prediction (Essentially tried it until we got one that was both accurate enough
        # and fast) also used Sequential because it was the most efficent (read it was the easiest)
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])

        # compile the code using adam optimizer (the one that consistently gave us the most accurate predictions) with a loss function
        # of mse (again the highest accuracy)
        model.compile(optimizer='adam', loss='mse')

        # Train the model (low epochs because it was taking a lot longer without it)
        model.fit(x_train, y_train, batch_size=1, epochs=1)


    # the test data using the days that weren't used for the training
    test = scaled_data[train_len - days:, :]

    # Create the data set x_test and y_test
    x_test = []
    y_test = dataset[train_len:, :]
    for i in range(days, len(test)):
        x_test.append(test[i-days:i, 0])

    # make it a numpy array
    x_test = np.array(x_test)

    # reshape data for the model to use
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)

    # convert back to USD
    predictions = scaler.inverse_transform(predictions)

    # get data we didnt touch
    valid = data[train_len:]
    
    # data that was predicted
    train = data[:train_len]
    valid['Predictions'] = predictions

    return train, valid, predictions, ticker, model

""" 
Plots the data
This function takes the data that was predicted and graphs in a chart using 
matplotlib.
"""

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

""" 
Main function
The main function that runs the predicion from the user
"""

def main():
    #remove pd and save model warnings
    pd.options.mode.chained_assignment = None
    absl.logging.set_verbosity(absl.logging.ERROR)

    #get ticker from user
    tix = input("What Stock ticker do you want to predict?\n")
    print("How strong do you want the model to be?")

    #get variance from user
    variance = input("Enter 0 for weak (short runtime), 1 for strong (long runtime)\n")
    while variance != 0 or variance != 1:
        if variance == 0 or variance == 1:
            break
        variance = input("This was not a valid input. Please enter 0, or 1\n")

    # fix variance
    if variance == 0:
        variance = .1
    elif variance == 1:
        variance = .02
    train, valid, predictions, ticker, model = prediction(tix, True)


    # make sure good start
    x = valid.head(1)
    real = x["Close"].values
    pred = x["Predictions"].values

    #repeat until data start is good, error checking
    while (real/pred) > 1+variance or (real/pred) < 1-variance:
        train, valid, predictions, ticker, model = prediction(tix)
        x = valid.head(1)
        real = x["Close"].values
        pred = x["Predictions"].values

    #save the model
    model.save(f"data/{ticker}")

    plot(train, valid, predictions, ticker)


if __name__ == "__main__":
    main()
