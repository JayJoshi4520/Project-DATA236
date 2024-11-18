from typing import Union
import yfinance as yf
from fastapi import FastAPI
import datetime
from pydantic import BaseModel
import json



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
    currDate = datetime.datetime.now()
    weekday_of_toay = datetime.datetime.today().weekday()
    if liveDataParams.interval == "1m" and liveDataParams.stockPeriod == "1D": 
        if  weekday_of_toay == 5:  # Saturday
            currDate = currDate - datetime.timedelta(days=1)
        elif weekday_of_toay == 6:  # Sunday
            currDate = currDate - datetime.timedelta(days=2)
    elif liveDataParams.interval == "1d" and liveDataParams.stockPeriod == "1W":
        currDate = currDate - datetime.timedelta(days=7)
    elif liveDataParams.interval == "1wk" and liveDataParams.stockPeriod == "1M":
        currDate = currDate - datetime.timedelta(days=30)
    elif liveDataParams.interval == "1mo" and liveDataParams.stockPeriod == "1Y":
        currDate = currDate - datetime.timedelta(days=365)
    formatted_date = currDate.strftime("%Y-%m-%d")

    try:
        data = yf.download(liveDataParams.ticker, start=formatted_date,interval=liveDataParams.interval, group_by=liveDataParams.ticker)
    except Exception as e:
        return {"Error": str(e)}
        
    formatted_data = []
    for date, row in data[liveDataParams.ticker].iterrows():
        prices = []
        entry = {
            "date": date.isoformat(),
            "high": row["High"],
            "volume": row["Volume"],
            "open": row["Open"],
            "low": row["Low"],
            "close": row["Close"],
            "adjClose": row["Adj Close"]
        }
        
        formatted_data.append(entry)
        

    stock_data = {
        "liveData": formatted_data,
        "currentPrice": formatted_data[-1]["adjClose"] if formatted_data else None,
        }

    
    return stock_data




