import mongoose, { Schema } from "mongoose";

const DatasetMetaSchema = new Schema(
  {
    name: { type: String, required: true },
    source: { type: String, required: true },
    filename: { type: String, required: true },
    collectionName: { type: String, required: true },
    fileSize: { type: Number, required: true },
    totalRecords: { type: Number, required: true },
    fileType: { type: String, required: true },
    status: {
      type: String,
      required: true,
      enum: ["raw", "latest", "preprocessing", "preprocessed", "validated", "ready", "archived", "failed"],
      default: "raw",
    },
    columns: { type: [String], required: true },
    description: { type: String },
    uploadDate: { type: Date, default: Date.now },
    errorMessage: { type: String },
    isAPI: { type: Boolean, default: false },
    lastUpdated: { type: Date },
    apiConfig: { type: Schema.Types.Mixed },
    deletedAt: { type: Date, default: null },
  },
  {
    collection: "dataset_meta",
  }
);

export const DatasetMeta = mongoose.models.DatasetMeta || mongoose.model("DatasetMeta", DatasetMetaSchema, "dataset_meta");
