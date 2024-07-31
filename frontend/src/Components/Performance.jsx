import { useEffect, useState } from "react";
import Navbar from "./Navbar";
import axios from "axios";
import Loader from "./loader";

const Performance = () => {
  const [data, setData] = useState(null);
  const [visibleRows, setVisibleRows] = useState(10); // State to control visible rows

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(
          "http://localhost:8080/api/portfolio/summary"
        );
        setData(response.data);
      } catch (error) {
        console.log("An error occurred", error);
      }
    };

    fetchData();
  }, []);

  // Function to load more rows
  const loadMoreRows = () => {
    setVisibleRows((prevVisibleRows) => prevVisibleRows + 10); // Increase visible rows by 10
  };

  return (
    <div className="container mx-auto">
      <Navbar />
      <div>
        <div className="overflow-x-auto mt-20 px-4">
          <div className="min-w-full bg-white shadow-md rounded-lg overflow-hidden">
            <div className="sticky top-0 bg-gray-200 p-4 text-xs md:text-sm font-semibold text-gray-700 z-10 grid grid-cols-10 gap-4">
              <div>
                <b>Stock Name</b>
              </div>
              <div>Return</div>
              <div>Risk</div>
              <div>SystRisk_var</div>
              <div>TotalRisk_var</div>
              <div>UnsystRisk_var</div>
              <div>Alpha</div>
              <div>Beta</div>
              <div>Capm_ret</div>
              <div>Sharpe Ratio</div>
            </div>

            <div className="divide-y divide-gray-200">
              {data ? (
                Object.keys(data)
                  .slice(0, visibleRows)
                  .map((stock, index) => (
                    <div
                      key={index}
                      className="grid grid-cols-10 gap-4 p-4 text-xs md:text-sm text-gray-600"
                    >
                      <div className="truncate">
                        <b>{stock}</b>
                      </div>
                      <div>{data[stock].Return.toFixed(2)}</div>
                      <div>{data[stock].Risk.toFixed(2)}</div>
                      <div>{data[stock].SystRisk_var.toFixed(2)}</div>
                      <div>{data[stock].TotalRisk_var.toFixed(2)}</div>
                      <div>{data[stock].UnsystRisk_var.toFixed(2)}</div>
                      <div>{data[stock].alpha.toFixed(2)}</div>
                      <div>{data[stock].beta.toFixed(2)}</div>
                      <div>{data[stock].capm_ret.toFixed(2)}</div>
                      <div>{data[stock].sharpe.toFixed(2)}</div>
                    </div>
                  ))
              ) : (
                <div className="text-center p-4">
                  <Loader />
                </div>
              )}
            </div>
            {data && visibleRows < Object.keys(data).length && (
              <div className="text-center p-4">
                <button
                  onClick={loadMoreRows}
                  className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                >
                  See More
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Performance;
