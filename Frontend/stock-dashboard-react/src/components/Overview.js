import React, { useContext, useEffect, useState } from "react";
import Card from "./Card";
import PredDataContext from "../context/PredDataContext";

const Overview = ({ symbol, price, change, changePercent, currency }) => {
  
  const { predData } = useContext(PredDataContext);
  const [showPrediction, setShowPrediction] = useState(() => {
    const localMode = localStorage.getItem("showPrediction");
    return localMode || 'false';
  });

  const [predPrice, setPredPrice] = useState()

  useEffect(() => {
    if (showPrediction === "true" && predData.length > 0) {
      setPredPrice(predData[predData.length - 1].open)
    }

    if(predPrice){
      console.log(predPrice)
      console.log(predData)
    }
  }, [showPrediction, predData]);

  return (
    <Card>
      <span className="absolute left-4 top-4 text-neutral-400 text-lg xl:text-xl 2xl:text-2xl">
        {symbol}
      </span>
      <div className="w-full h-full flex items-center justify-around mt-5">
        <span className="text-2xl xl:text-4xl 2xl:text-5xl flex items-center">
          {currency === "USD" ? `$` : ""} {showPrediction == "true" ? predPrice : price}
          <span className="text-lg xl:text-xl 2xl:text-2xl text-neutral-400 m-2">
            {currency}
          </span>
        </span>
        <span
          className={`text-lg xl:text-xl 2xl:text-2xl ${
            change > 0 ? "text-lime-500" : "text-red-500"
          }`}
        >
          {change} <span>({changePercent}%)</span>
        </span>
      </div>
    </Card>
  );
};

export default Overview;