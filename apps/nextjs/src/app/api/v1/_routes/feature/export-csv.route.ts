import { Hono } from "hono";
import db from "@/lib/database/db";
import mongoose from "mongoose";
import { DatasetMeta } from "@/lib/database/schema/feature/dataset-meta.model";
import { HoltWinterDaily } from "@/lib/database/schema/model/holt-winter.model";
import { convertToCSV } from "@/lib/export";
import { LSTMDaily } from "@/lib/database/schema/model/lstm.model";

export const exportRoute = new Hono();

// ────────────────
// 1. Export DatasetMeta Collection
// ────────────────
exportRoute.get("/dataset-meta", async (c) => {
  try {
    console.log("=== Export CSV via DatasetMeta ===");
    await db();

    const slug = c.req.query("category");
    const sortBy = c.req.query("sortBy") || "Date";
    const sortOrder = c.req.query("sortOrder") || "desc";

    if (!slug) return c.json({ message: "Missing collection name" }, 400);

    const meta = await DatasetMeta.findOne({ collectionName: slug }).lean<{ columns?: string[] }>();
    if (!meta) return c.json({ message: "Dataset not found" }, 404);

    let Model: mongoose.Model<any>;
    try {
      Model = mongoose.model(slug);
    } catch {
      Model = mongoose.model<any>(slug, new mongoose.Schema({}, { strict: false }), slug);
    }

    const sortQuery: Record<string, 1 | -1> = { [sortBy]: sortOrder === "desc" ? -1 : 1 };
    const data = await Model.find().sort(sortQuery).lean();

    const columns = (meta?.columns || Object.keys(data[0] || [])).map((col) => ({
      key: col,
      header: col,
      hasCustomCell: false,
    }));

    const csvContent = convertToCSV(data, columns);

    c.header("Content-Type", "text/csv;charset=utf-8;");
    c.header("Content-Disposition", `attachment; filename="${slug}_${new Date().toISOString().split("T")[0]}.csv"`);

    return c.text(csvContent);
  } catch (error) {
    console.error("Error exporting dataset-meta CSV:", error);
    return c.json({ message: "Failed to export dataset-meta CSV" }, 500);
  }
});

// ────────────────
// 2. Export Holt-Winter Collection
// ────────────────
exportRoute.get("/hw/daily", async (c) => {
  try {
    console.log("=== Export CSV via Holt-Winter Daily ===");
    await db();

    const sortBy = c.req.query("sortBy") || "forecast_date";
    const sortOrder = c.req.query("sortOrder") || "desc";

    const sortQuery: Record<string, 1 | -1> = { [sortBy]: sortOrder === "desc" ? -1 : 1 };
    const data = await HoltWinterDaily.find().sort(sortQuery).lean();

    const columns = data.length
      ? Object.keys(data[0]).map((col) => ({
          key: col,
          header: col,
          hasCustomCell: false,
        }))
      : [];

    const csvContent = convertToCSV(data, columns);

    c.header("Content-Type", "text/csv;charset=utf-8;");
    c.header("Content-Disposition", `attachment; filename="holt_winter_daily_${new Date().toISOString().split("T")[0]}.csv"`);

    return c.text(csvContent);
  } catch (error) {
    console.error("Error exporting HW CSV:", error);
    return c.json({ message: "Failed to export HW CSV" }, 500);
  }
});

exportRoute.get("/lstm/daily", async (c) => {
  try {
    console.log("=== Export CSV via LSTM Daily ===");
    await db();

    const sortBy = c.req.query("sortBy") || "forecast_date";
    const sortOrder = c.req.query("sortOrder") || "desc";

    const sortQuery: Record<string, 1 | -1> = { [sortBy]: sortOrder === "desc" ? -1 : 1 };
    const data = await LSTMDaily.find().sort(sortQuery).lean();

    if (!data || data.length === 0) {
      console.log("No LSTM data found for export");
      return c.json({ message: "No data found" }, 404);
    }

    // ✅ FIXED: Extract SEMUA parameters, bukan hanya yang pertama
    const csvData: any[] = [];
    
    data.forEach(item => {
      const params = item.parameters || {};
      
      // ✅ LOOP through SEMUA parameters dalam satu document
      Object.keys(params).forEach(paramKey => {
        const paramData = params[paramKey] || {};
        
        csvData.push({
          _id: item._id,
          forecast_date: new Date(item.forecast_date).toISOString().split('T')[0],
          timestamp: new Date(item.timestamp).toISOString(),
          config_id: item.config_id,
          model_type: item.model_type || 'LSTM',
          parameter_name: paramKey,  
          forecast_value: paramData.forecast_value || '',
          model: paramData.model_metadata?.model || '',
          lookback_days: paramData.model_metadata?.lookback_days || '',
          scaler: paramData.model_metadata?.scaler || '',
          epochs: paramData.model_metadata?.epochs || '',
          rmse: paramData.model_metadata?.rmse || '',
          mae: paramData.model_metadata?.mae || '',
          source_collection: item.source_collection || '',
          column_id: item.column_id || ''
        });
      });
    });

    
    const columns = [
      { key: '_id', header: 'ID', hasCustomCell: false },
      { key: 'forecast_date', header: 'Forecast Date', hasCustomCell: false },
      { key: 'timestamp', header: 'Created At', hasCustomCell: false },
      { key: 'config_id', header: 'Config ID', hasCustomCell: false },
      { key: 'model_type', header: 'Model Type', hasCustomCell: false },
      { key: 'parameter_name', header: 'Parameter', hasCustomCell: false },
      { key: 'forecast_value', header: 'Forecast Value', hasCustomCell: false },
      { key: 'model', header: 'Model Version', hasCustomCell: false },
      { key: 'lookback_days', header: 'Lookback Days', hasCustomCell: false },
      { key: 'scaler', header: 'Scaler Type', hasCustomCell: false },
      { key: 'epochs', header: 'Epochs', hasCustomCell: false },
      { key: 'rmse', header: 'RMSE', hasCustomCell: false },
      { key: 'mae', header: 'MAE', hasCustomCell: false },
      { key: 'source_collection', header: 'Source Collection', hasCustomCell: false },
      { key: 'column_id', header: 'Column ID', hasCustomCell: false }
    ];

    const csvContent = convertToCSV(csvData, columns);

    c.header("Content-Type", "text/csv;charset=utf-8;");
    c.header("Content-Disposition", `attachment; filename="lstm_forecast_${new Date().toISOString().split("T")[0]}.csv"`);

    return c.text(csvContent);
    
  } catch (error) {
    console.error("Error exporting LSTM CSV:", error);
    return c.json({ message: "Failed to export LSTM CSV", error: error instanceof Error ? error.message : String(error) }, 500);
  }
});

export default exportRoute;
