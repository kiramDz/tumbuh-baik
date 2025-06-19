import mongoose, { Schema } from "mongoose";

const BmkgDataSchema = new Schema(
  {
    Date: { type: Date, required: true },
    Year: { type: Number, required: true },
    Month: { type: String, required: true },
    Day: { type: Number, required: true },
    TN: { type: Number, required: true },
    TX: { type: Number, required: true },
    TAVG: { type: Number, required: true },
    RH_AVG: { type: Number, required: true },
    RR: { type: Number, required: true },
    SS: { type: Number, required: true },
    FF_X: { type: Number, required: true },
    DDD_X: { type: Number, required: true },
    FF_AVG: { type: Number, required: true },
    DDD_CAR: { type: String, required: true },
    Season: { type: String, required: true },
    is_RR_missing: { type: Number, required: true },
  },
  { timestamps: true }
);


export const BmkgData = mongoose.models.BmkgData || mongoose.model("BmkgData", BmkgDataSchema, "bmkg-data");
//                                     ^                          ^                       ^ (3rd param = nama collection persis)
