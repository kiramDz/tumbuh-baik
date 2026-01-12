import mongoose from "mongoose";

const UserSchema = new mongoose.Schema(
  {
    name: String,
    email: { type: String, unique: true },
    emailVerified: { type: Boolean, default: false },
    image: String,
    role: { type: String, default: "user", enum: ["user", "admin"] },
  },
  { timestamps: true }
);

export const User = mongoose.models.User || mongoose.model("User", UserSchema, "user");;
