import { Route, Routes } from "react-router-dom";
import "./App.css";
import Signup from "./Components/Signup";
import Home from "./Components/Home";
import { Login } from "./Components/Login";
import Prediction from "./Components/Prediction";
import Performance from "./Components/Performance";
function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Home />}></Route>
        <Route path="/signup" element={<Signup />}></Route>
        <Route path="/login" element={<Login />} />
        <Route path="/prediction" element={<Prediction />}></Route>
        <Route path="/performance" element={<Performance />}></Route>
      </Routes>
    </div>
  );
}

export default App;
