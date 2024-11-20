import axios from "axios";

const basePath = "https://finnhub.io/api/v1";
const baseUrl = "http://127.0.0.1:8000/getlivedata"
/**
 * Searches best stock matches based on a user's query
 * @param {string} query - The user's query, e.g. 'fb'
 * @returns {Promise<Object[]>} Response array of best stock matches
 */

export const searchSymbol = async (query) => {
  const url = `${basePath}/search?q=${query}&token=${process.env.REACT_APP_API_KEY}`;
  const response = await fetch(url);

  if (!response.ok) {
    const message = `An error has occured: ${response.status}`;
    throw new Error(message);
  }

  return await response.json();
};

/**
 * Fetches the details of a given company
 * @param {string} stockSymbol - Symbol of the company, e.g. 'FB'
 * @returns {Promise<Object>} Response object
 */
export const fetchStockDetails = async (stockSymbol) => {
  
  
  const url = `${basePath}/stock/profile2?symbol=${stockSymbol}&token=${process.env.REACT_APP_API_KEY}`;
  const response = await fetch(url)

  if (!response.ok) {
    const message = `An error has occured: ${response.status}`;
    throw new Error(message);
  }

  return await response.json();
};

/**
 * Fetches the latest quote of a given stock
 * @param {string} stockSymbol - Symbol of the company, e.g. 'FB'
 * @returns {Promise<Object>} Response object
 */
export const fetchQuote = async (stockSymbol) => {
  const url = `${basePath}/quote?symbol=${stockSymbol}&token=${process.env.REACT_APP_API_KEY}`;
  const response = await fetch(url);
  

  if (!response.ok) {
    const message = `An error has occured: ${response.status}`;
    throw new Error(message);
  }

  return await response.json();
};


export const fetchNews = async (category) => {
  const url = `${basePath}/news?category=${category}&token=${process.env.REACT_APP_API_KEY}`;
  const response = await fetch(url);
  

  if (!response.ok) {
    const message = `An error has occured: ${response.status}`;
    throw new Error(message);
  }

  return await response.json();
};

/**
 * Fetches historical data of a stock (to be displayed on a chart)
 * @param {string} stockSymbol - Symbol of the company, e.g. 'FB'
 * @param {string} resolution - Resolution of timestamps. Supported resolution includes: 1, 5, 15, 30, 60, D, W, M
 * @param {number} from - UNIX timestamp (seconds elapsed since January 1st, 1970 at UTC). Interval initial value.
 * @param {number} to - UNIX timestamp (seconds elapsed since January 1st, 1970 at UTC). Interval end value.
 * @returns {Promise<Object>} Response object
 */
export const fetchHistoricalData = async (
  stockSymbol,
  resolution,
  from,
  to
) => {
  const url = "http://127.0.0.1:8000/getlivedata"

  let interval = "1m"
  let stockPeriod = "1D"
  if(resolution === "1") {
    interval = "1m"
    stockPeriod = "1D"
  }else if(resolution === "15"){
    interval = "1d"
    stockPeriod = "1W"
  }else if(resolution === "60"){
    interval = "1wk"
    stockPeriod = "1M"
  }else if(resolution === "D"){
    interval = "1mo"
    stockPeriod = "1Y"
  }

  const data = {
    "ticker": stockSymbol,
    "interval": interval,
    "stockPeriod": stockPeriod
 }

 try {
  const response = await axios.post(url, data, {
    headers: {
      'Content-Type': 'application/json'
    }
  });
  
  return await response.data
} catch (error) {
  console.error('Error:', error);
}
};
