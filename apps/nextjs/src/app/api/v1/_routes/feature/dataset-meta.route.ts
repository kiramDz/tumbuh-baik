import { Hono } from "hono";
import db from "@/lib/database/db";
import { DatasetMeta } from "@/lib/database/schema/feature/dataset-meta.model";
import { parseError } from "@/lib/utils";
import mongoose from "mongoose";

const datasetMetaRoute = new Hono();

// GET - Ambil semua metadata dataset
datasetMetaRoute.get("/", async (c) => {
  try {
    await db();
    const datasets = await DatasetMeta.find({ deletedAt: null }) // 👈 hanya yg aktif
      .sort({ uploadDate: -1 })
      .lean();
    return c.json({ data: datasets }, 200);
  } catch (error) {
    console.error("Get dataset meta error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// GET - Display soft deleted datasets in recycle bin
datasetMetaRoute.get("/recycle-bin", async (c) => {
  try {
    await db();
    const page = Number(c.req.query("page")) || 1;
    const pageSize = Number(c.req.query("pageSize")) || 10;

    const total = await DatasetMeta.countDocuments({ deletedAt: { $ne: null } });
    console.log("Total deleted items:", total); // 👈 ad
    const datasets = await DatasetMeta.find({ deletedAt: { $ne: null } })
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .sort({ deletedAt: -1 })
      .lean();

    console.log("Found datasets:", datasets.length);

    return c.json({
      message: "Recycle bin data retrieved successfully",
      data: {
        items: datasets,
        total,
        currentPage: page,
        totalPages: Math.ceil(total / pageSize),
        pageSize,
      },
    });
  } catch (error) {
    console.error("Get recycle bin error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// GET - Buat slug untuk setiap dataset baru
datasetMetaRoute.get("/:slug", async (c) => {
  try {
    await db();
    const { slug } = c.req.param();
    console.log("[DEBUG] API dataset-meta called with slug:", slug);
    const page = Number(c.req.query("page")) || 1;
    const pageSize = Number(c.req.query("pageSize")) || 10;
    const sortBy = c.req.query("sortBy") || "Date";
    const sortOrder = c.req.query("sortOrder") || "desc";

    const meta = await DatasetMeta.findOne({ collectionName: slug }).lean();
    if (!meta) return c.json({ message: "Dataset not found" }, 404);

    let Model;
    try {
      Model = mongoose.model(slug);
    } catch {
      Model = (mongoose.models[slug] || mongoose.model(slug, new mongoose.Schema({}, { strict: false }), slug)) as mongoose.Model<any>;
    }

    const sortQuery: Record<string, 1 | -1> = {
      [sortBy]: sortOrder === "desc" ? -1 : 1,
    };
    const totalData = await Model.countDocuments();

    const data = await Model.find()
      .sort(sortQuery)
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .lean();

    return c.json(
      {
        message: "Success",
        data: {
          meta,
          items: data,
          total: totalData,
          currentPage: page,
          totalPages: Math.ceil(totalData / pageSize),
          pageSize,
          sortBy,
          sortOrder,
        },
      },
      200
    );
  } catch (error) {
    console.error("Error fetching dynamic dataset:", error);
    return c.json({ message: "Server error" }, 500);
  }
});

// Route untuk fetch semua data tanpa pagination (khusus chart)
datasetMetaRoute.get("/:slug/chart-data", async (c) => {
  try {
    await db();
    const { slug } = c.req.param();
    console.log("[DEBUG] API chart-data called with slug:", slug);

    // Gunakan type assertion untuk meta
    const meta = (await DatasetMeta.findOne({ collectionName: slug }).lean()) as {
      name: string;
      collectionName: string;
      columns: string[];
      [key: string]: any;
    } | null;

    if (!meta) return c.json({ message: "Dataset not found" }, 404);

    // Validasi kolom Date
    const hasDateColumn = meta.columns.some((col: string) => col.toLowerCase() === "date");

    if (!hasDateColumn) {
      return c.json(
        {
          message: "Dataset tidak memiliki kolom Date",
          data: null,
        },
        200
      );
    }

    let Model;
    try {
      Model = mongoose.model(slug);
    } catch {
      Model = (mongoose.models[slug] || mongoose.model(slug, new mongoose.Schema({}, { strict: false }), slug)) as mongoose.Model<any>;
    }

    // Fetch SEMUA data, sort by Date ascending untuk chart
    const allData = await Model.find().sort({ Date: 1 }).lean();

    // Filter hanya kolom numerik (exclude Date dan kolom string)
    const firstItem = allData[0] || {};
    const numericColumns = meta.columns.filter((col: string) => {
      if (col.toLowerCase() === "date") return false;
      const value = firstItem[col];
      return typeof value === "number" || !isNaN(Number(value));
    });

    return c.json(
      {
        message: "Success",
        data: {
          items: allData,
          numericColumns,
          dateColumn: "Date",
        },
      },
      200
    );
  } catch (error) {
    console.error("Error fetching chart data:", error);
    return c.json({ message: "Server error" }, 500);
  }
});
// DELETE - Soft delete dataset (move to recycle bin)
datasetMetaRoute.patch("/:collectionName/delete", async (c) => {
  try {
    await db();
    const { collectionName } = c.req.param();

    const dataset = await DatasetMeta.findOneAndUpdate(
      { collectionName },
      { $set: { deletedAt: new Date() } }, // 👈 set kolom deletedAt
      { new: true }
    );

    if (!dataset) return c.json({ message: "Dataset not found" }, 404);
    console.log("Updated dataset:", dataset);

    return c.json({ message: "Dataset moved to recycle bin", success: true, data: dataset }, 200);
  } catch (error) {
    console.error("Soft delete dataset error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// PATCH - Restore soft deleted dataset from recycle bin
datasetMetaRoute.patch("/:collectionName/restore", async (c) => {
  try {
    await db();
    const { collectionName } = c.req.param();

    const dataset = await DatasetMeta.findOneAndUpdate({ collectionName }, { deletedAt: null }, { new: true });

    if (!dataset) return c.json({ message: "Dataset not found" }, 404);

    return c.json({ message: "Dataset restored successfully", data: dataset }, 200);
  } catch (error) {
    console.error("Restore dataset error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// DELETE - Permanently delete dataset and its metadata (from recycle bin)
datasetMetaRoute.delete("/:collectionName", async (c) => {
  try {
    await db();
    const { collectionName } = c.req.param();

    // Hapus metadata
    await DatasetMeta.deleteOne({ collectionName });

    const connection = mongoose.connection;

    if (!connection.db) {
      throw new Error("Database connection is not established");
    }

    const collections = await connection.db.listCollections().toArray();
    const exists = collections.some((col) => col.name === collectionName);

    if (exists) {
      await connection.db.dropCollection(collectionName);
    }

    return c.json({ message: "Dataset deleted successfully" }, 200);
  } catch (error) {
    console.error("Delete dataset error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// POST - Upload dataset metadata + records
datasetMetaRoute.post("/", async (c) => {
  try {
    await db();
    const body = await c.req.json();

    const requiredFields = ["name", "source", "fileType", "data"];
    for (const field of requiredFields) {
      if (!body[field]) {
        return c.json({ message: `${field} is required` }, 400);
      }
    }

    if (!["csv", "json"].includes(body.fileType)) {
      return c.json({ message: "fileType must be 'csv' or 'json'" }, 400);
    }

    const {
      name,
      source,
      fileType,
      data, // data records (parsed JSON/CSV)
      filename = `${name}.${fileType}`,
      description = "",
      status = "raw",
      collectionName: rawCollectionName,
    } = body;

    if (!Array.isArray(data) || data.length === 0) {
      return c.json({ message: "data must be a non-empty array" }, 400);
    }

    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB
    const collectionName = rawCollectionName?.trim() || name.trim();
    const fileSize = Buffer.byteLength(JSON.stringify(body.data));

    if (fileSize > MAX_FILE_SIZE) {
      return c.json({ message: "File size exceeds 16MB limit" }, 400);
    }

    const totalRecords = data.length || 0;
    const columns = data[0] ? Object.keys(data[0]) : [];

    // Insert data ke collection dinamis
    const dynamicModel = mongoose.model(collectionName, new mongoose.Schema({}, { strict: false }), collectionName);
    await dynamicModel.insertMany(data);

    // Simpan metadata
    const newDataset = await DatasetMeta.create({
      name: name.trim(),
      source: source.trim(),
      filename,
      collectionName,
      fileType,
      status,
      description,
      fileSize,
      totalRecords,
      columns,
    });

    console.log({
      name,
      source,
      filename,
      collectionName,
      fileType,
      status,
      description,
      fileSize,
      totalRecords,
      columns,
    });

    return c.json(
      {
        message: "Dataset uploaded and metadata saved successfully",
        data: newDataset,
      },
      201
    );
  } catch (error) {
    console.error("Upload dataset error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

datasetMetaRoute.get("/rainfall-summary", async (c) => {
  try {
    await db();
    const Model = mongoose.models["rainfall"] || mongoose.model("rainfall", new mongoose.Schema({}, { strict: false }), "rainfall");

    const summary = await Model.aggregate([
      {
        $group: {
          _id: "$Year",
          avgRainfall: { $avg: "$RR_imputed" },
          minRainfall: { $min: "$RR_imputed" },
          maxRainfall: { $max: "$RR_imputed" },
          totalDays: { $sum: 1 },
        },
      },
      { $sort: { _id: 1 } },
      {
        $project: {
          year: "$_id",
          avgRainfall: 1,
          minRainfall: 1,
          maxRainfall: 1,
          totalDays: 1,
          _id: 0,
        },
      },
    ]);

    return c.json({ message: "Success", data: summary }, 200);
  } catch (error) {
    console.error("Error in rainfall-summary route:", error);
    return c.json({ message: "Server error" }, 500);
  }
});
export default datasetMetaRoute;