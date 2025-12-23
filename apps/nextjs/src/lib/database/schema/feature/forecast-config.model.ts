import mongoose, { Schema } from "mongoose";

const ForecastConfigSchema = new Schema(
  {
    name: { type: String, required: true },
    columns: [
      {
        collectionName: { type: String, required: true },
        columnName: { type: String, required: true },
      },
    ],
    status: { type: String, required: true, enum: ["pending", "running", "done", "failed"] },
    forecastResultCollection: { type: String },
    startDate: { type: Date },
    endDate: { type: Date },
    errorMessage: { type: String },
    error_metrics: [
      {
        collectionName: { type: String, required: true },
        columnName: { type: String, required: true },
        metrics: {
          mae: Number,
          r2: Number,
          mape: Number,
          mse: Number,
        },
      },
    ],
  },
  {
    collection: "forecast_configs",
    timestamps: true,
  }
);
export const ForecastConfig = mongoose.models.ForecastConfig || mongoose.model("ForecastConfig", ForecastConfigSchema, "forecast_configs");
