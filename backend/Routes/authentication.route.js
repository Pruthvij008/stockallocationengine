const express = require('express');
const router = express.Router();
const { signupController } = require('../Controllers/signupController');
const { loginController } = require('../Controllers/loginController');


router.post('/signup', signupController); 
router.post('/login', loginController); 

module.exports = router;
