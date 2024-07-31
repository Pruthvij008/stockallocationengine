const express = require("express");
const app = express();
const dbConnect = require('./Config/database');
require('dotenv').config();
const cookieParser = require('cookie-parser');
const cors = require('cors');

// Middlewares
app.use(express.json());
app.use(cookieParser());

// CORS configuration
app.use(cors({
  origin: "http://localhost:3000",
  credentials: true,
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowedHeaders: [
    "Content-Type",
    "Authorization",
    "Access-Control-Allow-Origin",
    "Access-Control-Allow-Credentials"
  ],
  optionsSuccessStatus: 200
}));

// Database connection
dbConnect();

// Routes
const authRoutes = require('./Routes/authentication.route');
const portfolioRoutes = require('./Routes/portfolio.route');

app.use('/api/auth', authRoutes); // Prefix API routes
app.use('/api/portfolio', portfolioRoutes); // Prefix API routes

// Home Route
app.get('/', (req, res) => {
  res.json({
    success: true,
    data: "Welcome to the home page",
    message: "This is the home page"
  });
});

// Server
const port = process.env.PORT ||8080; // Default to 5000 if no port in .env
app.listen(port, () => {
  console.log(`Listening on port ${port}`);
});
