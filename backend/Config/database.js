const mongoose = require("mongoose");

const dbConnect = async () => {
  try {
    const dbUri =
      process.env.MONGODB_URI || "mongodb://localhost:27017/yourdbname";

    await mongoose.connect(dbUri, {
      // useNewUrlParser: true,
      // useUnifiedTopology: true,
    });

    console.log("Connected to MongoDB");
  } catch (error) {
    console.error("Failed to connect to MongoDB:", error);
  }
};

module.exports = dbConnect;
