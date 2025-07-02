import mongoose, { Schema } from "mongoose";

const BuoysDataSchema = new Schema(
  {
    Date: { type: Date, required: true },
    Year: { type: Number, required: true },
    Month: { type: Number, required: true },
    Day: { type: Number, required: true },
    RAD: { type: Number, required: true },
    RAIN: { type: Number, required: true },
    RH: { type: Number, required: true },
    SST: { type: Number, required: true },
    TEMP_10_0m: { type: Number, required: true },
    TEMP_20_0m: { type: Number, required: true },
    TEMP_40_0m: { type: Number, required: true },
    TEMP_60_0m: { type: Number, required: true },
    TEMP_80_0m: { type: Number, required: true },
    TEMP_100_0m: { type: Number, required: true },
    TEMP_120_0m: { type: Number, required: true },
    TEMP_140_0m: { type: Number, required: true },
    TEMP_180_0m: { type: Number, required: true },
    TEMP_300_0m: { type: Number, required: true },
    TEMP_500_0m: { type: Number, required: true },
    UWND: { type: Number, required: true },
    VWND: { type: Number, required: true },
    WSPD: { type: Number, required: true },
    WDIR: { type: Number, required: true },
    Location: { type: String, required: true },
  },
  { timestamps: true }
);

export const BuoysData = mongoose.models.BuoysData || mongoose.model("BuoysData", BuoysDataSchema, "buoys");
