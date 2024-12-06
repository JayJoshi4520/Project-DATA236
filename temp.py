from typing import Union
import yfinance as yf
from fastapi import FastAPI
import datetime
from pydantic import BaseModel
import json
import pandas as pd
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
import keras
import warnings
import pandas as pd
import numpy as np


app = FastAPI()

class GetLiveData(BaseModel):
    ticker: str
    interval: str
    stockPeriod: str
    
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",  
    "https://your-frontend-domain.com",  
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          
    allow_credentials=True,
    allow_methods=["*"],           
    allow_headers=["*"],            
)




@app.post("/getlivedata")
def getLiveData(liveDataParams: GetLiveData):
    # Initialize current date
    currDate = datetime.datetime.now()
    formatted_date = currDate.strftime("%Y-%m-%d")
    data = None

    # Adjust date range based on interval and stock period
    if liveDataParams.interval == "1m" and liveDataParams.stockPeriod == "1D":
        # Go back one day at a time until data is available
        while data is None or data.empty:
            currDate -= datetime.timedelta(days=1)
            formatted_date = currDate.strftime("%Y-%m-%d")
            try:
                data = yf.download(
                    liveDataParams.ticker,
                    start=formatted_date,
                    interval=liveDataParams.interval,
                    group_by=liveDataParams.ticker
                )
            except Exception as e:
                return {"Error": str(e)}

    elif liveDataParams.interval == "1d" and liveDataParams.stockPeriod == "1W":
        currDate -= datetime.timedelta(days=7)
    elif liveDataParams.interval == "1wk" and liveDataParams.stockPeriod == "1M":
        currDate -= datetime.timedelta(days=30)
    elif liveDataParams.interval == "1mo" and liveDataParams.stockPeriod == "1Y":
        currDate -= datetime.timedelta(days=365)

    # Format the adjusted start date
    formatted_date = currDate.strftime("%Y-%m-%d")

    try:
        # Fetch data from yFinance
        data = yf.download(
            liveDataParams.ticker,
            start=formatted_date,
            interval=liveDataParams.interval,
            group_by=liveDataParams.ticker
        )
    except Exception as e:
        return {"Error": str(e)}

    if data.empty:
        return {"Error": "No data available for the given parameters."}

    # Format the data into a list of dictionaries
    formatted_data = []
    for date, row in data[liveDataParams.ticker].iterrows():
        entry = {
            "date": date,
            "high": row["High"],
            "volume": row["Volume"],
            "open": row["Open"],
            "low": row["Low"],
            "close": row["Close"],
            "adjClose": row["Adj Close"],
        }
        formatted_data.append(entry)


    # Build the response
    stock_data = {
        "liveData": formatted_data,
        "currentPrice": formatted_data[-1]["adjClose"] if formatted_data else None,
    }

    return stock_data




@app.post("/prediction")
def getPrediction(stockInfo: GetLiveData):
    currDate = datetime.datetime.now()
    weekday_of_toay = datetime.datetime.today().weekday()
    if stockInfo.interval == "1m" and stockInfo.stockPeriod == "1D": 
        if  weekday_of_toay == 5:  # Saturday
            currDate = currDate - datetime.timedelta(days=1)
        elif weekday_of_toay == 6:  # Sunday
            currDate = currDate - datetime.timedelta(days=2)
    elif stockInfo.interval == "1d" and stockInfo.stockPeriod == "1Y":
        currDate = currDate - datetime.timedelta(days=365)
    elif stockInfo.interval == "1wk" and stockInfo.stockPeriod == "1M":
        currDate = currDate - datetime.timedelta(days=30)
    elif stockInfo.interval == "1mo" and stockInfo.stockPeriod == "1Y":
        currDate = currDate - datetime.timedelta(days=365)
    formatted_date = currDate.strftime("%Y-%m-%d")

    # Calculating Bollinger Band
    while True:
        try:
            model = keras.models.load_model(f'./data_230/StockModel/LSTM/model_{stockInfo.ticker}.keras')
            break
        except:
            model_training(stockInfo.ticker)
            print("Error")
    data = yf.download(stockInfo.ticker, start="2010-01-01", interval=stockInfo.interval, group_by=stockInfo.ticker)
    close_prices = data[stockInfo.ticker]['Close']  
    MA20 = close_prices.rolling(window=20).mean()
    std_dev = close_prices.rolling(window=20).std()
    upper_band = MA20 + (std_dev * 2)
    lower_band = MA20 - (std_dev * 2)
    upper_band.fillna(method='bfill', inplace=True) 
    lower_band.fillna(method='bfill', inplace=True)

    #Calculating RSI 
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)


    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    avg_gain.fillna(method='bfill', inplace=True) 
    avg_loss.fillna(method='bfill', inplace=True)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))


    #Claculating Moving Average

        # Selecting the 'Close' for all tickers
    close_prices = pd.DataFrame(data[stockInfo.ticker]['Close'])

    lags = [5, 8, 13]
    for lag in lags:
        close_prices[f'{lag}-day EMA'] = close_prices['Close'].ewm(span=lag).mean()



    data = data.xs(key=stockInfo.ticker, level='Ticker', axis=1)['Close']
    data = pd.DataFrame(data)
    dataset = data.values

    training_data_len = int(np.ceil( len(dataset) * .95 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    
    LSTM_predictions = model.predict(x_test)
    LSTM_predictions = scaler.inverse_transform(LSTM_predictions)
    valid = data[training_data_len:]
    valid['LSTM_Predictions'] = LSTM_predictions
    result_array = [
    {"date": date.strftime('%Y-%m-%d'), "prediction": prediction}
    for date, prediction in zip(valid.index, valid.iloc[:, 0])
    ]

    data = yf.download(stockInfo.ticker, start="2010-01-01", interval=stockInfo.interval, group_by=stockInfo.ticker)
    data.loc[:, (str(stockInfo.ticker), 'Upper Band')] = upper_band
    data.loc[:, (str(stockInfo.ticker), 'Lower Band')] = lower_band
    data.loc[:, (str(stockInfo.ticker), '5-D EMA')] = close_prices['5-day EMA']
    data.loc[:, (str(stockInfo.ticker), '8-D EMA')] = close_prices['8-day EMA']
    data.loc[:, (str(stockInfo.ticker), '13-D EMA')] = close_prices['13-day EMA']
    formatted_data = []
    for date, row in data[stockInfo.ticker].iterrows():
        entry = {
            "high": row["High"],
            "volume": row["Volume"],
            "open": row["Open"],
            "low": row["Low"],
            "close": row["Close"],
            "adjClose": row["Adj Close"],
            "lower_band": row["Lower Band"],
            "upper_band": row["Upper Band"],
            "ma_5": row["5-D EMA"],
            "ma_8": row["8-D EMA"],
            "ma_13": row["13-D EMA"],
        }
        formatted_data.append(entry)
    rsi_data = []
    for date, row in rsi[stockInfo.ticker].iterrows():
        entry = {
            "high": row["High"],
            "volume": row["Volume"],
            "open": row["Open"],
            "low": row["Low"],
            "close": row["Close"],
            "adjClose": row["Adj Close"],
        }
        rsi_data.append(entry)

    return {
        "liveData": formatted_data,
        "prediction" : result_array,
        "rsi": rsi_data
    }



def model_training(stock_ticker):

    df = yf.download(stock_ticker, start="2010-01-01", interval="1d", group_by=stock_ticker)
    # Create a new dataframe with only the 'Close column 
    data = df.xs(key=stock_ticker, level='Ticker', axis=1)['Close']
    data = pd.DataFrame(data)
    dataset = data.values

    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    #Scaling Data Using MinMax Scaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set 
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]


    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            print(x_train)
            print(y_train)
            print()

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Building the GRU model
    GRU_MODEL = Sequential()
    GRU_MODEL.add(GRU(128, input_shape=(x_train.shape[1], 1)))
    GRU_MODEL.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Training LSTM(Long Short Term Memory - Neural Network) model
    print(f"Training LSTM Model {stock_ticker}: \n")
    model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data = [x_test, y_test])
    
    model.summary()

    
    
        # Compile the model
    GRU_MODEL.compile(optimizer='adam', loss='mean_squared_error')

    # Training GRU (Gated Recurrent Unit) Model
    print(f"Training GRU Model {stock_ticker}: \n")
    GRU_MODEL.fit(x_train, y_train, batch_size=128, epochs=30, validation_data = [x_test, y_test])

    # Printing model summary
    GRU_MODEL.summary()


    # Get the models predicted price values 
    LSTM_predictions = model.predict(x_test)
    GRU_predictions = GRU_MODEL.predict(x_test)
    LSTM_predictions = scaler.inverse_transform(LSTM_predictions)
    GRU_predictions = scaler.inverse_transform(GRU_predictions)
    
    model.save(f'./data_230/StockModel/LSTM/model_{stock_ticker}.keras')
    GRU_MODEL.save(f'./data_230/StockModel/GRU/model_{stock_ticker}.keras')
