import mongoose, { Schema } from "mongoose";

const DecompositionReportSchema = new Schema(
  {
    preprocessing_id: {
      type: Schema.Types.ObjectId,
      ref: "PreprocessingReport",
      required: true,
    },
    dataset_name: { type: String, required: true },
    dataset_type: { type: String, enum: ["nasa", "bmkg"], required: true },
    decomposition_method: { type: String, default: "STL" },
    timestamp: { type: Date, default: Date.now },
    parameters: { type: Schema.Types.Mixed },
  },
  {
    collection: "decomposition_report",
    timestamps: { createdAt: "timestamp", updatedAt: false },
  },
);

export const DecompositionReport =
  mongoose.models.DecompositionReport ||
  mongoose.model("DecompositionReport", DecompositionReportSchema);
