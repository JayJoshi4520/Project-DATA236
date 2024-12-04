import React, { useContext, useEffect, useState } from "react";
import ChartFilter from "./ChartFilter";
import Card from "./Card";
import {
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  AreaChart,
  Tooltip,
} from "recharts";
import ThemeContext from "../context/ThemeContext";
import StockContext from "../context/StockContext";
import { fetchHistoricalData, fetchPredData } from "../utils/api/stock-api";
import { createDate, convertDateToUnixTimestamp } from "../utils/helpers/date-helper";
import { chartConfig } from "../constants/config";

const Chart = () => {
  const { darkMode } = useContext(ThemeContext);
  const { stockSymbol } = useContext(StockContext);
  const [filter, setFilter] = useState(() => {
    const localStock = localStorage.getItem("graphFilter");
    return localStock || "1D";
  });
  const [data, setData] = useState([]);
  const [predData, setPredData] = useState([]);
  const [showPrediction, setShowPrediction] = useState(() => {
    const localMode = localStorage.getItem("showPrediction");
    console.log(localMode);
    return localMode || 'false';
  });

  const formatData = (liveData) => {
    return liveData.map((item) => ({
      value: parseFloat(item.open).toFixed(2),
      high: parseFloat(item.high).toFixed(2),
      close: parseFloat(item.close).toFixed(2),
      date: filter === "1D"
        ? new Date(item.date).toLocaleTimeString()
        : new Date(item.date).toLocaleDateString(),
    }));
  };

  const formatPredData = (predData) => {
    return predData.map((item) => ({
      open: parseFloat(item.prediction).toFixed(2),
      date: new Date(item.date).toLocaleDateString(),
    }));
  };

  useEffect(() => {
    const getDateRange = () => {
      const { days, weeks, months, years } = chartConfig[filter];
      const endDate = new Date();
      const startDate = createDate(endDate, -days, -weeks, -months, -years);
      return {
        startTimestampUnix: convertDateToUnixTimestamp(startDate),
        endTimestampUnix: convertDateToUnixTimestamp(endDate),
      };
    };

    const updateChartData = async () => {
      try {
        const { startTimestampUnix, endTimestampUnix } = getDateRange();
        const resolution = chartConfig[filter].resolution;

        const result = await fetchHistoricalData(stockSymbol, resolution, startTimestampUnix, endTimestampUnix);
        setData(formatData(result.liveData));

        if (showPrediction == "true") {
          const predictionResult = await fetchPredData(stockSymbol, resolution, startTimestampUnix, endTimestampUnix);
          setPredData(formatPredData(predictionResult.prediction));
        }
      } catch (error) {
        console.error("Error fetching data:", error);
        setData([]);
        setPredData([]);
      }
    };

    updateChartData();
  }, [stockSymbol, filter, showPrediction]);

  useEffect(() => {
    const handleStorageChange = () => {
      const localMode = localStorage.getItem("showPrediction");
      setShowPrediction(localMode === "true");
    };

    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);

  return (
    <Card>
      <ul className="flex absolute top-2 right-2 z-40">
        {showPrediction === "true" ? (
          <li key="single-item">
            <ChartFilter
              text={filter} // Ensure `item` is properly defined in the context
              active={filter}
            />
          </li>
        ) : (
          Object.keys(chartConfig).map((item) => (
            <li key={item}>
              <ChartFilter
                text={item}
                active={filter === item}
                onClick={() => {
                  setFilter(item);
                  localStorage.setItem("graphFilter", item);
                }}
              />
            </li>
          ))
        )}
      </ul>
      <ResponsiveContainer>
        <AreaChart data={showPrediction == "true" ? predData : data}>
          <defs>
            <linearGradient id="chartColor" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={showPrediction == "false" ? "#312e81" : darkMode ? "#db1212": "#9f0303"} stopOpacity={0.8} />
              <stop offset="95%" stopColor={showPrediction == "false" ? "#312e81" : darkMode ? "#312e81" : "#c7d2fe"} stopOpacity={0} />
            </linearGradient>
          </defs>
          <Tooltip
            content={({ active, payload, label }) =>
              active && payload && payload.length ? (
                <div
                  style={{
                    backgroundColor: darkMode ? "#111827" : "#fff",
                    color: darkMode ? "#818cf8" : "#000",
                    padding: "10px",
                    border: "1px solid #ccc",
                    borderRadius: "4px",
                  }}
                >
                  <p>{`Date: ${label}`}</p>
                  {Object.entries(payload[0].payload).map(([key, value]) =>
                    key !== "date" ? <p key={key}>{`${key}: ${value}`}</p> : null
                  )}
                </div>
              ) : null
            }
          />
          <Area
            type="monotone"
            dataKey={showPrediction == "true" ? "open" : "close"}
            stroke="#312e81"
            fill="url(#chartColor)"
            fillOpacity={1}
            strokeWidth={0.5}
          />
          <XAxis dataKey="date" />
          <YAxis domain={["dataMin", "dataMax"]} />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  );
};

export default Chart;
