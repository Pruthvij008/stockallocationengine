const userModel = require("../Models/user.model");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
require("dotenv").config();

const loginController = async (req, res) => {
  try {
    const { emailId, username, password } = req.body;
    console.log(username, emailId, password);

    if (!password || (!username && !emailId)) {
      return res.status(400).json({
        success: false,
        data: "Bad request",
        message: "Fields are empty",
      });
    }

    const user = await userModel.findOne({ $or: [{ emailId }, { username }] });
    console.log("User found:", user);

    if (!user) {
      return res.status(404).json({
        success: false,
        data: "User not found",
        message: "User does not exist",
      });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).json({
        success: false,
        data: "Login failed",
        message: "Invalid password",
      });
    }

    const payload = {
      userId: user._id,
      emailId: user.emailId,
    };

    const secretKey = process.env.JWT_KEY || "defaultSecretKey";
    const token = jwt.sign(payload, secretKey, { expiresIn: "5h" });

    user.password = undefined;

    const cookieOptions = {
      httpOnly: true,
      expires: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000),
    };

    res.cookie("token", token, cookieOptions).status(200).json({
      success: true,
      token,
      user,
      message: "User logged in successfully",
    });
  } catch (error) {
    console.error("An error occurred while logging in:", error);
    res.status(500).json({
      success: false,
      message: "An error occurred",
    });
  }
};

module.exports = { loginController };
