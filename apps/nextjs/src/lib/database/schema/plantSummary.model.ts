// src/lib/database/schema/plantSummary.model.ts
import mongoose, { Schema } from "mongoose";

const PlantSummarySchema = new Schema(
  {
    month: { type: String, required: true }, // format: YYYY-MM
    curah_hujan_total: { type: Number, required: true },
    kelembapan_avg: { type: Number, required: true },
    status: { type: String, required: true }, // contoh: "tidak cocok tanam"
    timestamp: { type: Date, required: true },
  },
  { timestamps: true }
);

export const PlantSummary = mongoose.models.PlantSummary || mongoose.model("PlantSummary", PlantSummarySchema, "bmkg-tanam-summary");
