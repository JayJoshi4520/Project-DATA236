import React, { useContext, useEffect, useState } from "react";
import ThemeIcon from "./ThemeIcon";
import ThemeContext from "../context/ThemeContext";

const TopHeader = ({ name, onPredictionChange }) => {
  const { darkMode } = useContext(ThemeContext);
  const [isOpen, setIsOpen] = useState(false);
  const [selectedOption, setSelectedOption] = useState(null);

  const options = [
    { value: "true", label: "Show Prediction: Yes" },
    { value: "false", label: "Show Prediction: No" },
  ];

  const toggleDropdown = () => {
    setIsOpen((prev) => !prev);
  };

  const handleOptionClick = (option) => {
    setSelectedOption(option);
    localStorage.setItem("showPrediction", option.value); // Store only the value
    setIsOpen(false);

    // Notify the parent component or update context
    onPredictionChange(option.value === "true");
  };

  useEffect(() => {
    try {
      const localShowPred = localStorage.getItem("showPrediction");
      if (localShowPred) {
        const matchedOption = options.find(
          (option) => option.value === localShowPred
        );
        if (matchedOption) {
          setSelectedOption(matchedOption);

          // Notify the parent or context on initial load
          onPredictionChange(localShowPred === "true");
        }
      }
    } catch (e) {
      console.error("Error retrieving showPrediction from localStorage", e);
    }
  }, [onPredictionChange]);

  return (
    <div
      className={`flex py-2 w-full items-center justify-center ${
        darkMode ? "bg-gray-900 text-gray-300" : "bg-neutral-100"
      }`}
    >
      <ul
        className={`flex lg:w-6/12 justify-around rounded-lg py-3 ${
          darkMode ? "bg-indigo-600 text-gray-300" : "bg-indigo-600"
        }`}
      >
        <li className="flex-1/2">
          <a
            className="text-center block rounded hover:border-gray-200 text-white hover:bg-gray-900 hover:text-gray-300 py-2 px-4"
            href="/"
          >
            Dashboard
          </a>
        </li>
        <li className="flex-1/2">
          <a
            className="text-center block rounded hover:border-gray-200 text-white hover:bg-gray-900 hover:text-gray-300 py-2 px-4"
            href="/blog"
          >
            Blog
          </a>
        </li>
      </ul>

      <div className="relative py-4 ml-4 w-2/12">
        <button
          onClick={toggleDropdown}
          className={`rounded-lg py-4 w-full h-full ${
            darkMode
              ? "bg-gray-800 text-gray-300"
              : "bg-gray-200 text-gray-700"
          } hover:bg-indigo-600 hover:text-white`}
        >
          {selectedOption ? selectedOption.label : "Select an option"}
        </button>
        {isOpen && (
          <ul
            className={`absolute mt-2 w-48 bg-white border rounded shadow-lg z-10 ${
              darkMode
                ? "bg-gray-800 text-gray-300 border-gray-700"
                : "bg-gray-200 text-gray-700 border-gray-300"
            }`}
          >
            {options.map((option) => (
              <li
                key={option.value}
                className={`px-4 py-2 cursor-pointer hover:bg-indigo-600 hover:text-white ${
                  darkMode ? "hover:bg-indigo-700" : "hover:bg-indigo-600"
                }`}
                onClick={() => handleOptionClick(option)}
              >
                {option.label}
              </li>
            ))}
          </ul>
        )}
      </div>

      <ThemeIcon />
    </div>
  );
};

export default TopHeader;
