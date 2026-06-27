const express = require("express");
const router = express.Router();
const { prediction, summary, meta } = require("../Controllers/predictionController");

router.post("/prediction", prediction);
router.get("/summary", summary);
router.get("/meta", meta);

module.exports = router;
