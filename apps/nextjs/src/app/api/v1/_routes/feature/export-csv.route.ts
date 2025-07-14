import { Hono } from "hono";
import db from "@/lib/database/db";
import mongoose from "mongoose";
import { DatasetMeta } from "@/lib/database/schema/feature/dataset-meta.model";
import { HoltWinterDaily } from "@/lib/database/schema/model/holt-winter.model";
import { convertToCSV } from "@/lib/export";

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
exportRoute.get("/hw-daily", async (c) => {
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

export default exportRoute;
