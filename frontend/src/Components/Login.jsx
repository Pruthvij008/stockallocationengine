import axios from "axios";
import { useState } from "react";

import { FaEye } from "react-icons/fa6";
import { FaEyeSlash } from "react-icons/fa6";
import { useNavigate } from "react-router-dom";

export const Login = () => {
  const Navigate = useNavigate();
  const [isVisible, setIsVisible] = useState(false);
  const [loginData, setLoginData] = useState({
    emailId: "",
    password: "",
  });
  const [isvalid, setIsvalid] = useState(true);

  const onChangeHandler = (e) => {
    setLoginData((prevData) => ({
      ...prevData,
      [e.target.name]: e.target.value,
    }));
  };
  const submitHandler = async () => {
    const response = await axios.post(
      "http://localhost:8080/api/authlogin",
      loginData
    );

    if (response.success == true) {
      Navigate("/");
    } else {
      setIsvalid(false);
    }
  };
  return (
    <div className=" flex justify-center items-center h-screen">
      <div className="bg-gray-200 p-8 rounded-md">
        <div class="font-bold text-3xl my-6 text-center text-black hover:text-green-700 transition duration-300 transform hover:scale-105">
          {" "}
          Login
        </div>

        <div className="username mb-4">
          <label htmlFor="emailId" className="font-bold block mb-2">
            Username:
          </label>
          <input
            type="text"
            name="userName"
            className="h-10 w-96 border border-black rounded-md text-center glassmorphism-container placeholder-black"
            placeholder="Enter your email id"
          />
        </div>
        <div className="password mb-4">
          <label
            htmlFor="password"
            className="font-bold flex justify-center  mr-2 mb-2"
          >
            Password:
          </label>

          <input
            type={isVisible ? "text" : "password"}
            name="password"
            className="h-10 w-96 border border-black rounded-md text-center glassmorphism-container placeholder-black"
            placeholder="Enter password"
          />
          <div
            className="flex justify-end cursor-pointer mt-2"
            onClick={() => setIsVisible(!isVisible)}
          >
            {isVisible ? <FaEye /> : <FaEyeSlash />}
          </div>
        </div>

        <button
          className="h-12 w-36   rounded-lg bg-customGreen-100 border-2 m-2 mb-8 font-bold"
          onClick={submitHandler}
        >
          Login
        </button>
        <div>
          <span>Don't have an account? </span>
          <span
            className="text-blue-600 font-bold cursor-pointer"
            onClick={() => Navigate("/signup")}
          >
            Signup
          </span>
        </div>

        <div className="invalid">
          {!isvalid && <p className="text-red-500">Invalid credentials!</p>}
        </div>
      </div>
    </div>
  );
};
