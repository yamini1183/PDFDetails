import axios from "axios";

const Api=axios.create({
    // baseURL: "http://localhost:8000", //for localhost connection with FastAPI backend 
    // baseURL:"https://aiplanet-fullstack-genrative-ai-app.onrender.com"   //connection with deployed backend deployed FastAPI app 
    baseURL: "https://pdfdetails.onrender.com/"
})

export default Api
