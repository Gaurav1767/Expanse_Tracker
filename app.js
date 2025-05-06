import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import userRoutes from "./routes/userRoutes.js";
import connectDB from "./config/db.js";
import cookieParser from "cookie-parser";
import transactionRoutes from "./routes/transactionRoutes.js";

const app = express();

// Connect to MongoDB
connectDB();

// Middleware

app.use(express.json());
app.use(cookieParser());
app.use(cors({ origin: "http://localhost:8080", credentials: true }));
app.use(bodyParser.json());
app.use("/api/users", userRoutes);
app.use("/api/transactions", transactionRoutes);

export default app;
