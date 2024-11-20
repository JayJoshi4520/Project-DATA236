import React from "react";
import Search from "./Search";
import ThemeIcon from "./ThemeIcon";

const Header = ({ name }) => {
  return (
    <>
      <div className="xl:px-32">
        <div>
          <ul class="flex">
            <li class="flex-1 mr-2">
              <a class="text-center block border border-blue-500 rounded py-2 px-4 bg-blue-500 hover:bg-blue-700 text-white" href="/">Dashboard</a>
            </li>
            <li class="flex-1 mr-2">
              <a class="text-center block border border-white rounded hover:border-gray-200 text-blue-500 hover:bg-gray-200 py-2 px-4" href="/blog">Blog</a>
            </li>
          </ul>
        </div>
        <div>
          <h1 className="text-5xl">{name}</h1>
          <Search />
        </div>
      </div>
      <ThemeIcon />
    </>
  );
};

export default Header;
