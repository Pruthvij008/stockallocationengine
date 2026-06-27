const express = require("express");
const app = express();
require("dotenv").config();
const cors = require("cors");

// Middlewares
app.use(express.json());

// CORS configuration
app.use(
  cors({
    origin: process.env.CLIENT_ORIGIN || "http://localhost:5173",
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
    optionsSuccessStatus: 200,
  })
);

// Routes
const portfolioRoutes = require("./Routes/portfolio.route");
app.use("/api/portfolio", portfolioRoutes);

// Home Route
app.get("/", (req, res) => {
  res.json({
    success: true,
    data: "Welcome to the home page",
    message: "This is the home page",
  });
});

// Server
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`Listening on port ${port}`);
});
