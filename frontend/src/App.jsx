import { Route, Routes } from "react-router-dom";
import "./App.css";
import Home from "./Components/Home";
import Prediction from "./Components/Prediction";
import Performance from "./Components/Performance";

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/prediction" element={<Prediction />} />
        <Route path="/performance" element={<Performance />} />
      </Routes>
    </div>
  );
}

export default App;
