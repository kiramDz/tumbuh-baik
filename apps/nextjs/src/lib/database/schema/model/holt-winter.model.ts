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

export const HoltWinterDaily = mongoose.models.HoltWinterDaily || mongoose.model("HoltWinterDaily", HoltWinterDailySchema, "holt-winter");
// export const HoltWinterDaily = mongoose.model("HoltWinterDaily", HoltWinterDailySchema, "holt-winter");
