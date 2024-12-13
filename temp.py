from typing import Union
import yfinance as yf
from fastapi import FastAPI, WebSocket
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
import asyncio
import requests
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()




# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GetLiveData(BaseModel):
    ticker: str
    timeframe: str
    # interval: str
    # stockPeriod: str
    
class ModelTraining(BaseModel):
    ticker: str

# @app.post("/getlivedata")
# async def get_live_data(request: dict):
#     try:
#         symbol = request.get('ticker', 'AAPL')
#         timeframe = request.get('timeframe', '1D')
        
#         stock = yf.Ticker(symbol)
        
#         # Define timeframe parameters
#         timeframe_params = {
#             '1D': ('1d', '1m'),
#             '1W': ('5d', '15m'),
#             '1M': ('1mo', '1h'),
#             '1Y': ('1y', '1d')
#         }
        
#         period, interval = timeframe_params.get(timeframe, ('1d', '1m'))
        
#         # Get historical data
#         hist = stock.history(period=period, interval=interval)
        
#         if hist.empty:
#             return {
#                 "success": False,
#                 "message": "No data available"
#             }
        
#         # Format the data
#         live_data = []
#         for index, row in hist.iterrows():
#             live_data.append({
#                 'date': index.strftime('%Y-%m-%d %H:%M:%S'),
#                 'close': float(row['Close']),
#                 'volume': int(row['Volume']),
#                 'high': float(row['High']),
#                 'low': float(row['Low']),
#                 'open': float(row['Open'])
#             })
        
#         # Get current price and calculate change
#         current_price = float(hist['Close'].iloc[-1])
#         prev_close = float(hist['Close'].iloc[0])
#         change = ((current_price - prev_close) / prev_close) * 100
        
#         return {
#             "success": True,
#             "symbol": symbol,
#             "currentPrice": current_price,
#             "change": change,
#             "liveData": live_data,
#             "timeframe": timeframe
#         }
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return {"success": False, "message": str(e)}

# @app.post("/getlivedata")
# async def get_live_data(request: dict):
#     try:
#         symbol = request.get('ticker', 'AAPL')
#         timeframe = request.get('timeframe', '1D')
        
#         stock = yf.Ticker(symbol)
        
#         # Define timeframe parameters
#         timeframe_params = {
#             '1D': ('1d', '1m'),
#             '1W': ('5d', '15m'),  # Changed from '1wk' to '5d'
#             '1M': ('1mo', '1h'),
#             '1Y': ('1y', '1d')
#         }
        
#         period, interval = timeframe_params.get(timeframe, ('1d', '1m'))
        
#         # Get historical data
#         hist = stock.history(period=period, interval=interval)
        
#         if hist.empty:
#             return {
#                 "success": False,
#                 "message": "No data available"
#             }
        
#         # Format the data and remove duplicates
#         live_data = []
#         seen_dates = set()
        
#         for index, row in hist.iterrows():
#             # For 1W view, use date only (without time) as the key
#             if timeframe == '1W':
#                 date_key = index.strftime('%Y-%m-%d')
#                 if date_key not in seen_dates:
#                     seen_dates.add(date_key)
#                     live_data.append({
#                         'date': index.strftime('%Y-%m-%d %H:%M:%S'),
#                         'close': float(row['Close']),
#                         'volume': int(row['Volume']),
#                         'high': float(row['High']),
#                         'low': float(row['Low']),
#                         'open': float(row['Open'])
#                     })
#             else:
#                 live_data.append({
#                     'date': index.strftime('%Y-%m-%d %H:%M:%S'),
#                     'close': float(row['Close']),
#                     'volume': int(row['Volume']),
#                     'high': float(row['High']),
#                     'low': float(row['Low']),
#                     'open': float(row['Open'])
#                 })
        
#         # Get current price and calculate change
#         current_price = float(hist['Close'].iloc[-1])
#         prev_close = float(hist['Close'].iloc[0])
#         change = ((current_price - prev_close) / prev_close) * 100
        
#         return {
#             "success": True,
#             "symbol": symbol,
#             "currentPrice": current_price,
#             "change": change,
#             "liveData": live_data,
#             "timeframe": timeframe
#         }
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return {"success": False, "message": str(e)}

# @app.post("/getlivedata")
# async def get_live_data(request: dict):
#     try:
#         symbol = request.get('ticker', 'AAPL')
#         timeframe = request.get('timeframe', '1D')
        
#         # Map META to its actual ticker symbol if needed
#         if symbol == 'META':
#             symbol = 'META'  # META is now the correct symbol, previously it was FB
            
#         stock = yf.Ticker(symbol)
        
#         # Define timeframe parameters
#         timeframe_params = {
#             '1D': ('1d', '1m'),
#             '1W': ('5d', '15m'),
#             '1M': ('1mo', '1h'),
#             '1Y': ('1y', '1d')
#         }
        
#         period, interval = timeframe_params.get(timeframe, ('1d', '1m'))
        
#         # Get historical data
#         hist = stock.history(period=period, interval=interval)
        
#         if hist.empty:
#             return {
#                 "success": False,
#                 "message": "No data available"
#             }
        
#         # Format the data
#         live_data = []
#         for index, row in hist.iterrows():
#             live_data.append({
#                 'date': index.strftime('%Y-%m-%d %H:%M:%S'),
#                 'close': float(row['Close']),
#                 'volume': int(row['Volume']),
#                 'high': float(row['High']),
#                 'low': float(row['Low']),
#                 'open': float(row['Open'])
#             })
        
#         # Get current price and calculate change
#         current_price = float(hist['Close'].iloc[-1])
#         prev_close = float(hist['Close'].iloc[0])
#         change = ((current_price - prev_close) / prev_close) * 100
        
#         return {
#             "success": True,
#             "symbol": symbol,
#             "currentPrice": current_price,
#             "change": change,
#             "liveData": live_data,
#             "timeframe": timeframe
#         }
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return {"success": False, "message": str(e)}

# @app.post("/getlivedata")
# async def get_live_data(request: dict):
#     try:
#         symbol = request.get('ticker', 'AAPL')
#         timeframe = request.get('timeframe', '1D')
        
#         stock = yf.Ticker(symbol)
#         info = stock.info  # Get detailed stock information
        
#         # Format market cap to be readable
#         def format_market_cap(market_cap):
#             if market_cap is None:
#                 return 'N/A'
#             if market_cap >= 1e12:
#                 return f'${market_cap/1e12:.2f}T'
#             if market_cap >= 1e9:
#                 return f'${market_cap/1e9:.2f}B'
#             if market_cap >= 1e6:
#                 return f'${market_cap/1e6:.2f}M'
#             return f'${market_cap:,.2f}'

#         # Get IPO date in a readable format
#         ipo_date = info.get('firstTradingDate')
#         if ipo_date:
#             from datetime import datetime
#             ipo_date = datetime.fromtimestamp(ipo_date).strftime('%Y-%m-%d')
        
#         # Get historical data for chart
#         timeframe_params = {
#             '1D': ('1d', '1m'),
#             '1W': ('5d', '15m'),
#             '1M': ('1mo', '1h'),
#             '1Y': ('1y', '1d')
#         }
        
#         period, interval = timeframe_params.get(timeframe, ('1d', '1m'))
#         hist = stock.history(period=period, interval=interval)
        
#         if hist.empty:
#             return {
#                 "success": False,
#                 "message": "No data available"
#             }
        
#         # Format the data
#         live_data = []
#         for index, row in hist.iterrows():
#             live_data.append({
#                 'date': index.strftime('%Y-%m-%d %H:%M:%S'),
#                 'close': float(row['Close']),
#                 'volume': int(row['Volume']),
#                 'high': float(row['High']),
#                 'low': float(row['Low']),
#                 'open': float(row['Open'])
#             })
        
#         # Calculate current price and change
#         current_price = float(hist['Close'].iloc[-1])
#         prev_close = float(hist['Close'].iloc[0])
#         change = ((current_price - prev_close) / prev_close) * 100
        
#         return {
#             "success": True,
#             "symbol": symbol,
#             "currentPrice": current_price,
#             "change": change,
#             "liveData": live_data,
#             "timeframe": timeframe,
#             # Add detailed stock information
#             "stockInfo": {
#                 "name": info.get('longName', symbol),
#                 "ipoDate": ipo_date or 'N/A',
#                 "country": info.get('country', 'N/A'),
#                 "marketCap": format_market_cap(info.get('marketCap')),
#                 "currency": info.get('currency', 'USD'),
#                 "industry": info.get('industry', 'N/A'),
#                 "sector": info.get('sector', 'N/A'),
#                 "exchange": info.get('exchange', 'N/A'),
#                 "website": info.get('website', 'N/A'),
#                 "longBusinessSummary": info.get('longBusinessSummary', 'N/A')
#             }
#         }
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return {"success": False, "message": str(e)}

@app.post("/getTicker")
def get_ticker(request: dict):
    # Define the Yahoo Finance search URL
    yfinance_url = "https://query2.finance.yahoo.com/v1/finance/search"
    
    # Set up headers and parameters for the request
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {
        "q": request.get('companyName'),
        "quotes_count": 1,
        "country": "United States"
    }
    
    # Make the GET request to Yahoo Finance
    response = requests.get(url=yfinance_url, params=params, headers={'User-Agent': user_agent})
    
    # Parse the JSON response
    data = response.json()
    
    # Extract and return the ticker symbol
    try:
        company_code = data['quotes'][0]['symbol']
        return {
            "ticker": company_code
        }
    except (KeyError, IndexError):
        return None




@app.post("/getlivedata")
async def get_live_data(request: dict):
    try:
        symbol = request.get('ticker')
        timeframe = request.get('timeframe')
        
        stock = yf.Ticker(symbol)
        companyName = stock.info['longName']
        
        def format_market_cap(market_cap):
            if market_cap is None or market_cap == 0:
                return 'N/A'
            if market_cap >= 1e12:
                return f'${market_cap/1e12:.2f}T'
            if market_cap >= 1e9:
                return f'${market_cap/1e9:.2f}B'
            if market_cap >= 1e6:
                return f'${market_cap/1e6:.2f}M'
            return f'${market_cap:,.2f}'

        
        # Get historical data
        timeframe_params = {
            '1D': ('1d', '1m'),
            '1W': ('5d', '15m'),
            '1M': ('1mo', '1h'),
            '1Y': ('1y', '1d')
        }
        
        period, interval = timeframe_params.get(timeframe)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return {
                "success": False,
                "message": "No data available"
            }
        
        # Format the data
        live_data = []
        for index, row in hist.iterrows():
            live_data.append({
                'date': int(pd.Timestamp(index).timestamp()),
                'close': float(row['Close']),
                'volume': int(row['Volume']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'open': float(row['Open'])
            })
        
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[0])
        change = ((current_price - prev_close) / prev_close) * 100
        if symbol == "TSLA":
            print(current_price)
        
        return {
            "success": True,
            "company": companyName,
            "symbol": symbol,
            "currentPrice": current_price,
            "change": change,
            "liveData": live_data,
            "timeframe": timeframe,
        }
        
    except Exception as e:
        print(f"Error fetching data for {symbol}:", str(e))
        return {"success": False, "message": str(e)}

@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    await websocket.accept()
    
    try:
        while True:
            # Get real-time data
            stock = yf.Ticker(symbol)
            current = stock.history(period='1d', interval='1m')
            
            if not current.empty:
                current_price = float(current['Close'].iloc[-1])
                prev_close = float(current['Open'].iloc[0])
                change = ((current_price - prev_close) / prev_close) * 100
                
                data = {
                    "success": True,
                    "data": {
                        "price": current_price,
                        "change": change,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                await websocket.send_text(json.dumps(data))
            
            await asyncio.sleep(1)  # Update every second
            
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")
    finally:
        await websocket.close()




@app.post("/prediction")
def getPrediction(stockInfo: GetLiveData):
    stock = yf.Ticker(stockInfo.ticker)
    companyName = stock.info['longName']
    timeframe_params = {
            '1D': ('1d', '1m'),
            '1W': ('5d', '15m'),
            '1M': ('1mo', '1h'),
            '1Y': ('1y', '1d')
        }
    
    print(stockInfo.ticker, stockInfo.timeframe)
    period, interval = timeframe_params.get(stockInfo.timeframe)
    model = keras.models.load_model(f'./data_230/StockModel/LSTM/model_{stockInfo.ticker}.keras')

    data = yf.download(stockInfo.ticker, start="2019-01-01", interval=interval, group_by=stockInfo.ticker)

    formatted_data = []
    for date, row in data[stockInfo.ticker].iterrows():
        timestamp_obj = pd.Timestamp(date)
        unix_timestamp = int(timestamp_obj.timestamp())
        entry = {
            "date" : unix_timestamp,
            "high": row["High"],
            "volume": row["Volume"],
            "open": row["Open"],
            "low": row["Low"],
            "close": row["Close"],
            "adjClose": row["Adj Close"],
        }
        formatted_data.append(entry)

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
    {"date": int(pd.Timestamp(date).timestamp()), "prediction": prediction}
    for date, prediction in zip(valid.index, valid.iloc[:, 0])
    ]
    return {
        "company":companyName,
        "liveData": formatted_data,
        "prediction" : result_array,
    }


@app.post("/training")
def model_training(stock_ticker: ModelTraining):
    df = yf.download(stock_ticker.ticker, start="2019-01-01", interval="1d", group_by=stock_ticker.ticker)
    # Create a new dataframe with only the 'Close column 
    data = df.xs(key=stock_ticker.ticker, level='Ticker', axis=1)['Close']
    data = pd.DataFrame(data)
    dataset = data.values

    training_data_len = int(np.ceil( len(dataset) * .95 ))

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
    model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data = [x_test, y_test])
    
    model.summary()

    # Get the models predicted price values 
    LSTM_predictions = model.predict(x_test)
    LSTM_predictions = scaler.inverse_transform(LSTM_predictions)
    
    model.save(f'./data_230/StockModel/LSTM/model_{stock_ticker.ticker}.keras')
    return {
        "status": True
    }

