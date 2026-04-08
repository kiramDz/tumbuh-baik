import mongoose, { Schema } from "mongoose";

const PreprocessingReportSchema = new Schema(
  {
    dataset_type: { type: String, enum: ["nasa", "bmkg"], required: true },
    original_collection_name: { type: String, required: true },
    cleaned_collection_name: { type: String, required: true },
    preprocessing_timestamp: { type: Date, default: Date.now },
    preprocessing_summary: { type: Schema.Types.Mixed },
    quality_metrics: { type: Schema.Types.Mixed },
    imputation_validation: { type: Schema.Types.Mixed },
    smoothing_validation: { type: Schema.Types.Mixed },
    model_coverage: { type: Schema.Types.Mixed },
    warnings: { type: [String] },
    status: { type: String },
    record_count: {
      original: { type: Number },
      processed: { type: Number },
    },
  },
  {
    collection: "preprocessing_report",
    timestamps: { createdAt: "preprocessing_timestamp", updatedAt: false },
  },
);

export const PreprocessingReport =
  mongoose.models.PreprocessingReport ||
  mongoose.model("PreprocessingReport", PreprocessingReportSchema);
