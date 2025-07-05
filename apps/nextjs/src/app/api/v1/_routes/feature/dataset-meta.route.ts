import { Hono } from "hono";
import db from "@/lib/database/db";
import { DatasetMeta } from "@/lib/database/schema/feature/dataset-meta.model";
import { parseError } from "@/lib/utils";
import mongoose from "mongoose";

const datasetMetaRoute = new Hono();

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
