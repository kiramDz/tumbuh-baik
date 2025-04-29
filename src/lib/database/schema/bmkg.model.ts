import mongoose, { Schema } from "mongoose";

const BmkgDataSchema = new Schema(
  {
    city: { type: String, required: true },
    cloud_coverage: { type: Number, required: true },
    feels_like: { type: Number, required: true },
    humidity: { type: Number, required: true },
    lat: { type: Number, required: true },
    lon: { type: Number, required: true },
    pressure: { type: Number, required: true },
    rain_1h: { type: Number, required: true },
    sunrise: { type: String, required: true },
    sunset: { type: String, required: true },
    temperature: { type: Number, required: true },
    timestamp: { type: String, required: true },
    visibility: { type: mongoose.Schema.Types.Mixed, required: true }, // bisa string / number
    weather_description: { type: String, required: true },
    weather_main: { type: String, required: true },
    wind_deg: { type: Number, required: true },
    wind_gust: { type: Number, required: true },
    wind_speed: { type: Number, required: true },
  },
  { timestamps: true }
);

export const BmkgData = mongoose.models.BmkgData || mongoose.model("BmkgData", BmkgDataSchema, "bmkg-data");
//                                     ^                          ^                       ^ (3rd param = nama collection persis)
