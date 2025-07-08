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
    const datasets = await DatasetMeta.find().sort({ uploadDate: -1 }).lean();
    return c.json({ data: datasets }, 200);
  } catch (error) {
    console.error("Get dataset meta error:", error);
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

export default datasetMetaRoute;
