import React, { useContext, useState } from "react";
import StockContext from "../context/StockContext";
import ThemeContext from "../context/ThemeContext";

const SearchResults = ({ results }) => {
  const { darkMode } = useContext(ThemeContext);
  const { setStockSymbol } = useContext(StockContext);

  const [isVisible, setIsVisible] = useState(true);

  const handleOptionClick = (symbol) => {
    setStockSymbol(symbol);
    localStorage.setItem("ticker", symbol);
    setIsVisible(false); 
  };

  return (
    isVisible && (
      <ul
        className={`absolute top-12 border-2 w-full rounded-md h-64 overflow-y-scroll ${
          darkMode
            ? "bg-gray-900 border-gray-800 custom-scrollbar custom-scrollbar-dark"
            : "bg-white border-neutral-200 custom-scrollbar"
        }`}
      >
        {results.map((item) => (
          <li
            key={item.symbol}
            className={`cursor-pointer p-4 m-3 flex items-center justify-between rounded-md ${
              darkMode ? "hover:bg-indigo-600" : "hover:bg-indigo-200"
            } transition duration-300`}
            onClick={() => handleOptionClick(item.symbol)}
          >
            <span>{item.symbol}</span>
            <span>{item.description}</span>
          </li>
        ))}
      </ul>
    )
  );
};

export default SearchResults;
