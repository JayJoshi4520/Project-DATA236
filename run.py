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
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np


def getPrediction(ticker, interval, stockPeriod):
    currDate = datetime.datetime.now()
    weekday_of_toay = datetime.datetime.today().weekday()
    if interval == "1m" and stockPeriod == "1D": 
        if  weekday_of_toay == 5:  # Saturday
            currDate = currDate - datetime.timedelta(days=1)
        elif weekday_of_toay == 6:  # Sunday
            currDate = currDate - datetime.timedelta(days=2)
    elif interval == "1d" and stockPeriod == "1Y":
        currDate = currDate - datetime.timedelta(days=365)
    elif interval == "1wk" and stockPeriod == "1M":
        currDate = currDate - datetime.timedelta(days=30)
    elif interval == "1mo" and stockPeriod == "1Y":
        currDate = currDate - datetime.timedelta(days=365)
    formatted_date = currDate.strftime("%Y-%m-%d")

    data = yf.download(ticker, start="2010-01-01",interval=interval, group_by=ticker)
    data = data.xs(key=ticker, level='Ticker', axis=1)['Close']
    data = pd.DataFrame(data)
    model = keras.models.load_model(f'../data_230/StockModel/LSTM/model_{ticker}.keras')
    dataset = data.values
    print(dataset)

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
    print(valid)

    
getPrediction("AAPL", "1d", "1Y") 
