import { Hono } from "hono";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";
import { convertToCSV } from "@/lib/export";
import mongoose from "mongoose";
import { DatasetMeta } from "@/lib/database/schema/feature/dataset-meta.model";

const exportRoute = new Hono();

exportRoute.get("/", async (c) => {
  try {
    console.log("=== Export CSV Endpoint terpanggil ===");
    await db();

    const slug = c.req.query("category"); // sebenarnya ini adalah collectionName
    const sortBy = c.req.query("sortBy") || "Date";
    const sortOrder = c.req.query("sortOrder") || "desc";

    if (!slug) {
      return c.json({ message: "Missing collection name" }, 400);
    }

    const meta = await DatasetMeta.findOne({ collectionName: slug }).lean<{ columns?: string[] }>();
    if (!meta) return c.json({ message: "Dataset not found" }, 404);

    let Model: mongoose.Model<any>;
    try {
      Model = mongoose.model(slug);
    } catch {
      Model = mongoose.model<any>(slug, new mongoose.Schema({}, { strict: false }), slug);
    }

    const sortQuery: Record<string, 1 | -1> = {
      [sortBy]: sortOrder === "desc" ? -1 : 1,
    };

    const data = await Model.find().sort(sortQuery).lean();
    const columns = (meta?.columns || Object.keys(data[0] || [])).map((col) => ({
      key: col,
      header: col, // Atau bisa di-prettify kalau ingin
      hasCustomCell: false,
    }));

    const csvContent = convertToCSV(data, columns);

    c.header("Content-Type", "text/csv;charset=utf-8;");
    c.header("Content-Disposition", `attachment; filename="${slug}_${new Date().toISOString().split("T")[0]}.csv"`);

    return c.text(csvContent);
  } catch (error) {
    console.error("Error exporting CSV:", error);
    return c.json(
      {
        message: "Error",
        description: parseError(error),
        data: null,
      },
      { status: 500 }
    );
  }
});

export default exportRoute;
