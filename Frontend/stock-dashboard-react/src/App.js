import { useEffect, useState } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "./components/Dashboard";
import StockContext from "./context/StockContext";
import ThemeContext from "./context/ThemeContext";
import Header from "./components/TopHeader";
import Blog from "./pages/Blog";
import { PredDataProvider } from "./context/PredDataContext";

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const localMode = localStorage.getItem("mode");
    return localMode === "true";
  });

  const [stockSymbol, setStockSymbol] = useState(() => {
    const localStock = localStorage.getItem("ticker");
    return localStock || "AAPL";
  });

  const [showPrediction, setShowPrediction] = useState(true);

  useEffect(() => {
    try {
      let localMode = localStorage.getItem("mode");
      let localStock = localStorage.getItem("ticker");

      if (!localStock) {
        setStockSymbol("AAPL");
      }

      if (localMode === null) {
        setDarkMode(false);
      } else {
        setDarkMode(localMode === "true");
      }
    } catch (e) {
      console.log(e);
    }
  }, []);

  return (
    <ThemeContext.Provider value={{ darkMode, setDarkMode }}>
      <StockContext.Provider value={{ stockSymbol, setStockSymbol }}>
        <PredDataProvider>
        <Header onPredictionChange={setShowPrediction}/>
        <BrowserRouter>
          <Routes>
              <Route path="/" element={<Dashboard showPrediction={showPrediction} />} />
              <Route path="/blog" element={<Blog />} />
          </Routes>
        </BrowserRouter>
        </PredDataProvider>
      </StockContext.Provider>
    </ThemeContext.Provider>
  );
}

export default App;
