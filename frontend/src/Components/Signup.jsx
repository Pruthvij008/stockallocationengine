import { useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { useNavigate } from "react-router-dom";
import { FaEye } from "react-icons/fa6";
import { FaEyeSlash } from "react-icons/fa6";
export default function Signup() {
  const Navigate = useNavigate();

  const [formData, setFormData] = useState({
    username: "",
    password: "",
    emailId: "",
  });
  const [isvalid, setIsvalid] = useState(true);

  const [isVisible, setIsVisible] = useState(false);
  const [isVisible2,setIsVisible2]=useState(false);
  const handleOnChange = (e) => {
    setFormData((prevData) => ({
      ...prevData,
      [e.target.name]: e.target.value,
    }));
  };

  const invalidNotification = () => {
    console.log("this is toast function that hasn't called");
    toast.error("Error Notification !", {
      position: toast.POSITION.TOP_CENTER,
    });
  };
  const submitHandler = async () => {
    try {
      const response = await axios.post("http://localhost:8080/api/auth/signup", formData);
      if (response.success==true) {
        Navigate("/");
      } else {
        setIsvalid(false);

        invalidNotification();
        console.log(formData);
      }
    } catch (err) {
      console.log("An error occured while Signing up");
    }
  };

  return (
    <div className="flex h-screen justify-center items-center">
      <div className="bg-gray-200 shadow-lg rounded-md flex flex-col items-center p-8 w-1/2 max-w-lg">
        <div className="font-bold text-3xl my-6">Sign up</div>
        
        <div className="w-full mb-6">
          <label htmlFor="username" className="font-bold block mb-2">Username:</label>
          <input
            onChange={handleOnChange}
            type="text"
            placeholder="Enter your name"
            name="username"
            className="h-12 w-full rounded-lg text-center"
            value={formData.userName}
          />
        </div>

        <div className="w-full mb-6">
          <label htmlFor="email" className="font-bold block mb-2">Email Id:</label>
          <input
            onChange={handleOnChange}
            type="email"
            name="emailId"
            placeholder="abc@gmail.com"
            className="h-12 w-full rounded-lg text-center"
          />
        </div>

        <div className="w-full mb-6 relative">
          <label htmlFor="password" className="font-bold block mb-2">Password:</label>
          <input
            onChange={handleOnChange}
            type={isVisible ? "text" : "password"}
            name="password"
            placeholder="Password"
            className="h-12 w-full rounded-lg text-center"
          />
          <span className="absolute right-4 top-12 cursor-pointer" onClick={() => setIsVisible(!isVisible)}>
            {isVisible ? <FaEyeSlash /> : <FaEye />}
          </span>
        </div>

        <div className="w-full mb-6 relative">
          <label htmlFor="confirmpassword" className="font-bold block mb-2">Confirm password:</label>
          <input
            onChange={handleOnChange}
            type={isVisible2 ? "text" : "password"}
            name="confirmpassword"
            placeholder="Confirm password"
            className="h-12 w-full rounded-lg text-center"
            value={formData.confirmPassword}
          />
          <span className="absolute right-4 top-12 cursor-pointer" onClick={() => setIsVisible2(!isVisible2)}>
            {isVisible2 ? <FaEyeSlash /> : <FaEye />}
          </span>
        </div>

        <button className="h-12 w-36 mb-4 rounded-lg bg-customGreen-100  " onClick={submitHandler}>Sign up</button>

        <div className="mb-4">
          <span>Already a user? </span>
          <span className="text-blue-600 font-bold cursor-pointer" onClick={() => Navigate('/login')}>Login</span>
        </div>

        <div className="invalid">
          {!isvalid && <p className="text-red-500">Invalid credentials!</p>}
        </div>
      </div>

      <ToastContainer />
    </div>
  );
}
