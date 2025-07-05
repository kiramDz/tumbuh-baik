import mongoose, { Schema } from "mongoose";

const DatasetMetaSchema = new Schema(
  {
    name: { type: String, required: true }, // Nama koleksi, digunakan untuk identifikasi
    source: { type: String, required: true },
    filename: { type: String, required: true }, // Nama file yang diupload
    fileType: { type: String, required: true },
    filePath: { type: String, required: true },
    status: { type: String, required: true },
    description: { type: String },
    collectionTarget: { type: String, required: true },
    month: { type: String, required: true },
    timestamp: { type: Date, required: true },
  },
  {
    timestamps: true,
    collection: "dataset_meta",
  }
);

export const DatasetMeta = mongoose.models.DatasetMeta || mongoose.model("DatasetMeta", DatasetMetaSchema, "dataset_meta");
