import { createContext, useState } from 'react';

const PredDataContext = createContext();

export const PredDataProvider = ({ children }) => {
  const [predData, setPredData] = useState([]);

  return (
    <PredDataContext.Provider value={{ predData, setPredData }}>
      {children}
    </PredDataContext.Provider>
  );
};

export default PredDataContext;