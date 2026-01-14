import mongoose, { Schema } from "mongoose";

const LSTMDailySchema = new Schema(
  {
    forecast_date: { type: Date, required: true },
    timestamp: { type: Date, required: true },
    config_id: { type: String, required: true },
    parameters: { type: Schema.Types.Mixed, required: true },
  },
  { collection: "lstm-forecast", timestamps: false }
);

export const LSTMDaily = mongoose.models.LSTMDaily || mongoose.model("LSTMDaily", LSTMDailySchema, "lstm-forecast");
