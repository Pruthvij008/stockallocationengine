import { Route, Routes } from "react-router-dom";
import "./App.css";
import Home from "./Components/Home";
import Prediction from "./Components/Prediction";
import Performance from "./Components/Performance";
import Backtest from "./Components/Backtest";

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/prediction" element={<Prediction />} />
        <Route path="/performance" element={<Performance />} />
        <Route path="/backtest" element={<Backtest />} />
      </Routes>
    </div>
  );
}

export default App;
