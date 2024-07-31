const mongoose=require('mongoose');
const  userSchema=mongoose.Schema({

    username:{
        type:String,
        required:true,
        maxLength:200
    },
    emailID:{
        type:String,
        required:true,
        maxLength:100
    },
    password:{
        type:String,
        required:true,
        maxLength:100
    },
    age:{
        type:Number,
        required:false
    },
    portfolio:{
        type:mongoose.Schema.Types.ObjectId,
        ref:'Portfolio'
    }
})

module.exports=mongoose.model('user',userSchema);