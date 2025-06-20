import mongoose, { Schema } from "mongoose";

const SeedSchema = new Schema(
  {
    name: { type: String, required: true },
    duration: { type: Number, required: true }, // dalam hari
  },
  { timestamps: true } // mencatat createdAt dan updatedAt
);

export const Seed = mongoose.models.Seed || mongoose.model("Seed", SeedSchema, "seeds");
