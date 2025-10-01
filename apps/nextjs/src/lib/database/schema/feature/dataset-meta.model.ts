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
    status: { type: String, required: true },
    columns: { type: [String], required: true },
    description: { type: String },
    uploadDate: { type: Date, default: Date.now },
    errorMessage: { type: String },
    deletedAt: { type: Date, default: null },
  },
  {
    collection: "dataset_meta",
  }
);

export const DatasetMeta = mongoose.models.DatasetMeta || mongoose.model("DatasetMeta", DatasetMetaSchema, "dataset_meta");

// Yang manual diisi user: name, source, collectionName (optional), description (optional) - sudah benar.
