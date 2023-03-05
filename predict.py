from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from polygon import RESTClient
from polygon.rest import models
import pandas as pd
import yfinance as yahoo
import os.path


def prediction(ticker, useOldModel = False):
    currtime = datetime.now().strftime('%Y-%m-%d')
    print(currtime, type(currtime))
    df = None
    if not os.path.exists("data"):
        os.mkdir("data")
        
    if os.path.isfile(f"data/{ticker}.csv"):
        dateOfFile = datetime.fromtimestamp(os.path.getmtime(f"data/{ticker}.csv")).strftime('%Y-%m-%d')
        
        if dateOfFile < currtime:
            print("Fetching new Data")
            pd.DataFrame(yahoo.download(ticker, "2012-01-01", currtime)).to_csv(f"data/{ticker}.csv")
    else:
        print("Getting new Data")
        pd.DataFrame(yahoo.download(ticker, "2012-01-01", currtime)).to_csv(f"data/{ticker}.csv")

    df = pd.read_csv(f"data/{ticker}.csv")

    plt.style.use("dark_background")

    data = df.filter(["Close"])
    dataset = data.values

    train_split = .80

    train_len = math.ceil(len(dataset) * train_split)

    days = 60

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Checks if a model is saved
    if useOldModel and os.path.isfile(f"data/{ticker}/saved_model.pb"):

        # Check that the model exists
        model = load_model(f"data/{ticker}")

    else :

        
        train_data = scaled_data[0:train_len, :]

        x_train = []
        y_train = []

        for i in range(days, len(train_data)):
            x_train.append(train_data[i-days:i, 0])
            y_train.append(train_data[i,0])
                
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        #model.add(Dense(25))
        model.add(Dense(1))

        #compile
        model.compile(optimizer='adam', loss='mean_squared_error')

        #fit using x,y,every piece of data,only 1 run
        model.fit(x_train, y_train, batch_size=1, epochs=1)


    test = scaled_data[train_len - days: , :]

    #Create the data set x_test and y_test
    x_test = []
    y_test = dataset[train_len:, :]
    for i in range(days, len(test)):
        x_test.append(test[i-days:i, 0])

    #make it a numpy array
    x_test = np.array(x_test)

    #reshape data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #Get the models predicted price values
    predictions = model.predict(x_test)
    model.save(f"data/{ticker}")
    predictions = scaler.inverse_transform(predictions)


    #Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    rmse
    print(rmse)

    #get data we didnt touch
    valid = data[train_len:]
    #data that was predicted
    train = data[:train_len]
    valid['Predictions'] = predictions

    #plot data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
    plt.savefig('image.png')

    print(predictions[-1])

    print(valid)

def main():
    prediction('AAPL', True)

main()
