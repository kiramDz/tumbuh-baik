import mongoose, { Schema } from "mongoose";

const LSTMConfigSchema = new Schema(
    {
        name: { type: String, required: true },
        columns: [
            {
                collectionName: { type: String, required: true },
                columnName: { type: String, required: true }
            },
        ],
        status: { type: String, required: true, enum: ["pending", "running", "done", "failed"] },
        forecastResultCollection: { type: String },
        startDate: { type: Date },
        endDate: { type: Date },
        errorMessage: { type: String },
        error_metrics: [
            {
                collectionName: { type: String, required: true },
                columnName: { type: String, required: true },
                metrics: {
                    mae: Number,
                    rmse: Number,
                    aic: Number,
                    mse: Number,
                    mape: Number
                }
            }
        ]
    },
    {
        collection: "lstm_configs",
        timestamps: true
    }
);
export const LSTMConfig = mongoose.models.LSTMConfig || mongoose.model("LSTMConfig", LSTMConfigSchema, "lstm_configs");