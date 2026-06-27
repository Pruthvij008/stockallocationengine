const mongoose = require("mongoose");

// A single optimized allocation result tied to the user's inputs.
const holdingSchema = new mongoose.Schema(
  {
    ticker: { type: String, required: true },
    weight: { type: Number, required: true }, // percentage (0-100)
  },
  { _id: false }
);

const portfolioSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    investmentAmount: { type: Number, required: true },
    investmentPeriod: { type: Number, required: true },
    riskTolerance: {
      type: String,
      enum: ["low", "medium", "high"],
      required: true,
    },
    expectedReturn: { type: Number },
    holdings: [holdingSchema],
    sharpeRatio: { type: Number },
    annualizedReturn: { type: Number },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Portfolio", portfolioSchema);
