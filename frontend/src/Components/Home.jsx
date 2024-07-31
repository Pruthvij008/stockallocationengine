import { useNavigate } from "react-router-dom";
import Navbar from "./Navbar";
const Home = () => {
  const Navigate = useNavigate();
  console.log("this page has been called");
  return (
    <div className="min-h-screen bg-gray-900 text-white overflow-clip">
      <Navbar />
      <main className="flex items-center justify-center p-8">
        <img
          src={require("../Components/stockprediction.jpeg")}
          alt="Main visual"
          className="w-1/2 rounded-lg shadow-lg"
        />
        <div className="ml-8 p-8 bg-customGreen-100 rounded-lg">
          <h1 className="text-4xl font-bold">
            Invest in the freedom to choose
          </h1>
          <p className="mt-4">
            Wealth is not just about money. It's about what all you can do with
            it. It is having your own story of progress. And living it every
            single day. So go ahead, imagine a future you want to shape.
          </p>
          <p className="mt-2 font-bold">And make it happen.</p>
          <button
            className="mt-4 px-4 py-2 bg-black text-white rounded hover:bg-gray-800"
            onClick={() => {
              Navigate("/prediction");
            }}
          >
            Get started
          </button>
        </div>
      </main>
    </div>
  );
};

export default Home;
