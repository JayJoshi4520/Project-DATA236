import React, { useContext } from "react";
import Search from "./Search";
import ThemeIcon from "./ThemeIcon";
import ThemeContext from "../context/ThemeContext";

const Header = ({ name }) => {
  const { darkMode } = useContext(ThemeContext);
  return (
    <>
      <div className="xl:px-32 w-full">
        <div>
          <h1 className="text-5xl">{name}</h1>
          <Search />
        </div>
      </div>
    </>
  );
};

export default Header;
