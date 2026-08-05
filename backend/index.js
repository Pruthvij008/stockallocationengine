const express = require("express");
const app = express();
require("dotenv").config();
const cors = require("cors");

// Render (and similar hosts) inject cross-service URLs as bare hostnames, e.g.
// "investify-ml.onrender.com". Prepend https:// so the value is a usable URL.
const withScheme = (value) => {
  if (!value) return value;
  const v = value.trim();
  return /^https?:\/\//i.test(v) ? v : `https://${v}`;
};

// Normalise the Flask URL BEFORE the routes/controllers read it at import time.
if (process.env.FLASK_SERVER_URL) {
  process.env.FLASK_SERVER_URL = withScheme(process.env.FLASK_SERVER_URL);
}

// Middlewares
app.use(express.json());

// CORS configuration — allow the deployed frontend origin (and localhost in dev).
const clientOrigin = withScheme(process.env.CLIENT_ORIGIN) || "http://localhost:5173";
app.use(
  cors({
    origin: clientOrigin,
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
    optionsSuccessStatus: 200,
  })
);

// Health check — cheap, no downstream calls; used by Render's health probe and
// by the keep-alive pinger.
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", service: "express", time: new Date().toISOString() });
});

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

// Keep-alive: on a free hosting tier, web services sleep after ~15 min of no
// traffic. When running on Render (RENDER_EXTERNAL_URL is set), ping our own
// health endpoint and the Flask service every 14 minutes so both stay warm.
// A single external uptime monitor hitting /api/health is the robust backup.
const KEEPALIVE_MS = 14 * 60 * 1000;
const selfUrl = process.env.RENDER_EXTERNAL_URL;
if (selfUrl) {
  const ping = (url) =>
    fetch(url)
      .then(() => console.log(`keep-alive ok: ${url}`))
      .catch((e) => console.log(`keep-alive failed: ${url} (${e.message})`));
  setInterval(() => {
    ping(`${withScheme(selfUrl)}/api/health`);
    if (process.env.FLASK_SERVER_URL) ping(`${process.env.FLASK_SERVER_URL}/health`);
  }, KEEPALIVE_MS);
}
