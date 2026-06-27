const mongoose = require("mongoose");

const userSchema = new mongoose.Schema(
  {
    username: {
      type: String,
      required: true,
      trim: true,
      maxLength: 200,
    },
    emailId: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
      maxLength: 100,
    },
    password: {
      type: String,
      required: true,
      maxLength: 100,
    },
    age: {
      type: Number,
      required: false,
    },
    portfolio: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Portfolio",
    },
  },
  { timestamps: true }
);

module.exports = mongoose.model("User", userSchema);
