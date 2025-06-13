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

const DailySummarySchema = new Schema(
  {
    forecast_date: { type: Date, required: true },
    timestamp: { type: Date, required: true },
    parameters: {
      RR: {
        forecast_value: Number,
        model_metadata: {
          alpha: Number,
          beta: Number,
          gamma: Number,
        },
      },
      RH_AVG: {
        forecast_value: Number,
        model_metadata: {
          alpha: Number,
          beta: Number,
          gamma: Number,
        },
      },
    },
  },
  { timestamps: false }
);

export const PlantSummary = mongoose.models.PlantSummary || mongoose.model("PlantSummary", PlantSummarySchema, "bmkg-tanam-summary");
export const DailySummary = mongoose.models.DailySummary || mongoose.model("DailySummary", DailySummarySchema, "bmkg-hw");