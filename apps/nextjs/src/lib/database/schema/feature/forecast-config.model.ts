import mongoose, { Schema } from "mongoose";

const ForecastConfigSchema = new Schema(
  {
    name: { type: String, required: true }, // ✅ baru
    collectionName: { type: String, required: true },
    columnName: { type: String, required: true },
    status: {
      type: String,
      required: true,
      enum: ["pending", "running", "done", "failed"],
    },
    forecastResultCollection: { type: String },
    errorMessage: { type: String },
  },
  {
    collection: "forecast_configs",
    timestamps: true,
  }
);
  

export const ForecastConfig = mongoose.models.ForecastConfig || mongoose.model("ForecastConfig", ForecastConfigSchema, "forecast_configs");
