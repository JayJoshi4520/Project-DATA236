import React, { useContext } from "react";
import ThemeIcon from "./ThemeIcon";
import ThemeContext from "../context/ThemeContext";

const TopHeader = ({ name }) => {
  const { darkMode } = useContext(ThemeContext);
  return (
    <>
        <div className={`flex py-2 w-full items-center justify-center ${darkMode ? "bg-gray-900 text-gray-300" : "bg-neutral-100"}`}>
          <ul 
          className={`flex lg:w-6/12 justify-around rounded-lg py-3 ${darkMode ? "bg-indigo-600 text-gray-300" : "bg-indigo-600"}` }
          >
            <li className="flex-1/2">
              <a className="text-center block rounded hover:border-gray-200 text-white hover:bg-gray-900 hover:text-gray-300 py-2 px-4" href="/">Dashboard</a>
            </li>
            <li className="flex-1/2">
              <a className="text-center block rounded hover:border-gray-200 text-white hover:bg-gray-900 hover:text-gray-300 py-2 px-4" href="/blog">Blog</a>
            </li>
          </ul>
            <ThemeIcon />
        </div>
    </>
  );
};

export default TopHeader;