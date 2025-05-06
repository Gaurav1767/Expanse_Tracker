import express from "express";
import {
  registerUser,
  loginUser,
  getUserProfile,
  getUserProfileById,
  logoutUser,
} from "../controllers/userControllers.js";

const router = express.Router();

router.post("/signup", registerUser);
router.post("/login", loginUser);
router.get("/profile", getUserProfile);
router.get("/:userId", getUserProfileById);
router.post("/logout", logoutUser);

export default router;
