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


    data = yf.download(stockInfo.ticker, start="2010-01-01", interval=stockInfo.interval, group_by=stockInfo.ticker)
    close_prices = data[stockInfo.ticker]['Close']  
    MA20 = close_prices.rolling(window=20).mean()
    std_dev = close_prices.rolling(window=20).std()
    upper_band = MA20 + (std_dev * 2)
    lower_band = MA20 - (std_dev * 2)
    # Handle NaN values
    upper_band.fillna(method='bfill', inplace=True) 
    lower_band.fillna(method='bfill', inplace=True)

    data = data.xs(key=stockInfo.ticker, level='Ticker', axis=1)['Close']
    data = pd.DataFrame(data)
    model = keras.models.load_model(f'./data_230/StockModel/LSTM/model_{stockInfo.ticker}.keras')
    dataset = data.values

    training_data_len = int(np.ceil( len(dataset) * .95 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
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
    data.loc[:, ('AAPL', 'Upper Band')] = upper_band
    data.loc[:, ('AAPL', 'Lower Band')] = lower_band
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
            "upper_band": row["Upper Band"]
        }
        formatted_data.append(entry)

    return {
        "liveData": formatted_data,
        "prediction" : result_array
    }
