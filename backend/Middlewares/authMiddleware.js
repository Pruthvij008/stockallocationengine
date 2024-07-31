const jwt = require('jsonwebtoken');

exports.authMiddleware = (req, res, next) => {
    const token = req.cookies.token;

    if (!token) {
        return res.status(401).json({
            success: false,
            message: "Authorization token not provided"
        });
    }

    jwt.verify(token, process.env.JWT_KEY || "defaultSecretKey", (err, decoded) => {
        if (err) {
            console.error("Token verification failed:", err);
            return res.status(403).json({
                success: false,
                message: "Failed to authenticate token",
                error: err.message
            });
        }

        req.userId = decoded.userId;
        next();
    });
};
