import mongoose, { Schema } from "mongoose";

const HoltWinterDailySchema = new Schema(
  {
    forecast_date: { type: Date, required: true },
    timestamp: { type: Date, required: true },
    config_id: { type: String, required: true },
    parameters: { type: Schema.Types.Mixed, required: true },
  },
  { collection: "holt-winter", timestamps: false }
);



const HoltWinterSummarySchema = new Schema(
  {
    month: { type: String, required: true },
    kt_period: { type: String, required: true },
    status: { type: String, required: true }, // cocok, rehat, dsb
    reason: { type: String, required: true },
    parameters: { type: Schema.Types.Mixed, required: true },
    config_id: { type: String, required: true },
    location: { type: String, required: true },
  },
  { collection: "holt-winter-summary", timestamps: true }
);

export const HoltWinterSummary = mongoose.models.HoltWinterSummary || mongoose.model("HoltWinterSummary", HoltWinterSummarySchema, "holt-winter-summary");
export const HoltWinterDaily = mongoose.models.HoltWinterDaily || mongoose.model("HoltWinterDaily", HoltWinterDailySchema, "holt-winter");
