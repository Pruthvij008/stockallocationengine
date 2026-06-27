const express = require("express");
const router = express.Router();
const { prediction, summary, meta, backtest } = require("../Controllers/predictionController");

router.post("/prediction", prediction);
router.get("/summary", summary);
router.get("/meta", meta);
router.get("/backtest", backtest);

module.exports = router;
