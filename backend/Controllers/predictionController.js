const axios = require("axios");
require("dotenv").config();
const url = process.env.FLASK_SERVER_URL;

if (!url) {
  console.error(
    "FLASK_SERVER_URL is not set — every call to the ML service will fail. " +
      "Set it to the Flask service's URL (e.g. https://investify-ml.onrender.com)."
  );
}

// Distinguishes "not configured" and "upstream unreachable" from a genuine
// error returned by the ML service, so a misconfigured deploy is obvious.
const upstreamError = (error, fallback) => {
  if (!url) return "ML service URL is not configured (FLASK_SERVER_URL)";
  if (error.response) return error.response.data?.error || fallback;
  return `Could not reach the ML service at ${url}`;
};

exports.prediction = async (req, res) => {
  try {
    const {
      investment_amount,
      investment_period,
      risk_tolerance,
      expected_return,
      num_stocks,
    } = req.body;

    if (!investment_amount || !investment_period || !risk_tolerance) {
      return res.status(400).json({
        success: false,
        data: null,
        message: "Fields are empty",
      });
    }

    const data = {
      investment_amount,
      investment_period,
      risk_tolerance,
      expected_return,
      num_stocks,
    };

    // Send POST request to Flask server and await response
    const response = await axios.post(`${url}/predict`, data);

    // Return the response data to the client
    return res.status(200).json({
      success: true,
      data: response.data,
      message: "Data fetched successfully",
    });
  } catch (error) {
    console.error("An error occurred while predicting the stock:", error);
    return res.status(500).json({
      success: false,
      data: null,
      message: "An error occurred while predicting the stock",
    });
  }
};

exports.meta = async (req, res) => {
  try {
    const metaResponse = await axios.get(`${url}/meta`);
    res.json(metaResponse.data);
  } catch (error) {
    console.error("Error occurred while fetching meta:", error.message);
    res.status(500).json({ error: upstreamError(error, "Failed to fetch data coverage") });
  }
};

exports.backtest = async (req, res) => {
  try {
    // Pass through query params (top_k, model) to the Flask ML service.
    // The backtest runs in the background there, so preserve the 202
    // "still computing" status — the client polls until it gets a 200.
    const response = await axios.get(`${url}/backtest`, { params: req.query });
    res.status(response.status).json(response.data);
  } catch (error) {
    console.error("Error occurred while running the backtest:", error.message);
    res.status(500).json({ error: upstreamError(error, "Failed to run the backtest") });
  }
};

exports.summary = async (req, res) => {
  try {
    const summaryResponse = await axios.get(`${url}/summary`);

    console.log("Summary response data:", summaryResponse.data);
    res.json(summaryResponse.data);
  } catch (error) {
    console.error("Error occurred while fetching the summary:", error.message);
    res.status(500).json({ error: upstreamError(error, "Internal server error") });
  }
};
