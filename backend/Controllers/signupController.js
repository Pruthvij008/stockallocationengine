const userModel = require('../Models/user.model');
const bcrypt = require('bcrypt');

const signupController = async (req, res) => {
    try {
        const { username, password, age, emailId } = req.body;
        console.log(username, password, age, emailId);

        if (!emailId || !password || !username) {
            return res.status(400).json({
                success: false,
                data: "Bad request",
                message: "Fields are empty"
            });
        }

        const existingUser = await userModel.findOne({ emailId });
        console.log("User found: " + existingUser);

        if (existingUser) {
            return res.status(409).json({
                success: false,
                data: "User already exists",
                message: "User already exists"
            });
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        const newUser = await userModel.create({
            username,
            emailId,
            password: hashedPassword,
            age
        });

        newUser.password = undefined;

        res.status(201).json({
            success: true,
            data: newUser,
            message: 'Profile created successfully'
        });
    } catch (error) {
        console.error("An error occurred while signing up:", error);
        res.status(500).json({
            success: false,
            message: "An error occurred"
        });
    }
};

module.exports = { signupController };
