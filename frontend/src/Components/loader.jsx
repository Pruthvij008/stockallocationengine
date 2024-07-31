// Loader.js
import React from "react";

const Loader = () => {
  return (
    <div className="flex items-center justify-center space-x-2">
      <div className="spinner-border animate-spin inline-block w-8 h-8 border-4 rounded-full text-blue-500 border-t-transparent border-blue-500"></div>
      <span>Loading...</span>
    </div>
  );
};

export default Loader;
