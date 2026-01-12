import mongoose, { Schema } from "mongoose";

export interface IDatasetMeta extends Document {
  name: string;
  source: string;
  filename: string;
  collectionName: string;
  fileSize: number;
  totalRecords: number;
  fileType: string;
  status: string;
  columns: string[];
  description?: string;
  uploadDate: Date;
  errorMessage?: string;
  deletedAt?: Date | null;
}

const DatasetMetaSchema = new Schema(
  {
    name: { type: String, required: true }, // Nama koleksi, digunakan untuk identifikasi
    source: { type: String, required: true },
    filename: { type: String, required: true }, // Nama file yang diupload
    collectionName: { type: String, required: true },
    fileSize: { type: Number, required: true },
    totalRecords: { type: Number, required: true },
    fileType: { type: String, required: true },
    status: {
      type: String,
      required: true,
      enum: [
        "raw",
        "latest",
        "preprocessing",
        "preprocessed",
        "validated",
        "ready",
        "archived",
        "failed",
      ],
      default: "raw",
    },
    columns: { type: [String], required: true },
    description: { type: String },
    uploadDate: { type: Date, default: Date.now },
    errorMessage: { type: String },
    isAPI: { type: Boolean, default: false }, // Flag untuk menandakan dataset fetch vs upload
    lastUpdated: { type: Date },
    apiConfig: { type: Schema.Types.Mixed }, // Optional: menyimpan konfigurasi API jika diperlukan
    deletedAt: { type: Date, default: null }, // Untuk soft delete
  },
  {
    collection: "dataset_meta",
  }
);

export const DatasetMeta =
  mongoose.models.DatasetMeta ||
  mongoose.model("DatasetMeta", DatasetMetaSchema, "dataset_meta");
