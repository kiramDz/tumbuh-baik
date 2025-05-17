import mongoose, { Schema } from "mongoose";

const BMKGApiSchema = new Schema(
  {
    kode_gampong: { type: String, required: true },
    nama_gampong: { type: String, required: true },
    tanggal_data: { type: String, required: true }, // format: YYYY-MM-DD
    analysis_date: { type: Date, required: true },
    data: [
      {
        local_datetime: { type: String, required: true },
        t: { type: Number, required: true }, // suhu (°C)
        hu: { type: Number, required: true }, // kelembapan (%)
        weather_desc: { type: String, required: true }, // kondisi cuaca
        ws: { type: Number, required: true }, // kecepatan angin (km/jam)
        wd: { type: String, required: true }, // arah angin
        tcc: { type: Number, required: true }, // tutupan awan (%)
        vs_text: { type: String, required: true }, // jarak pandang (km)
      },
    ],
  },
  { timestamps: true }
);

export const BMKGApi = mongoose.models.BMKGApi || mongoose.model("BMKGApi", BMKGApiSchema, "bmkg-api");
