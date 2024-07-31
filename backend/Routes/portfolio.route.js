const express = require("express");
const router = express.Router();
const { prediction, summary } = require("../Controllers/predictionController");
const { authMiddleware } = require("../Middlewares/authMiddleware");

router.post("/prediction", prediction);
router.get("/summary",  summary);

module.exports = router;
