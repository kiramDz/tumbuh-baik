// routes/dataset.route.ts
import { Hono } from "hono";
import db from "@/lib/database/db";
import { DatasetMeta } from "@/lib/database/schema/feature/dataset-meta.model";
import { parseError } from "@/lib/utils";

const datasetMetaRoute = new Hono();

// POST - Upload dataset metadata
datasetMetaRoute.post("/", async (c) => {
  try {
    await db();
    const body = await c.req.json();

    const requiredFields = ["name", "source", "filename", "fileType", "collectionTarget", "month", "timestamp"];
    for (const field of requiredFields) {
      if (!body[field]) {
        return c.json({ message: `${field} is required` }, 400);
      }
    }

    if (!["csv", "json"].includes(body.fileType)) {
      return c.json({ message: "fileType must be 'csv' or 'json'" }, 400);
    }

    const newDataset = await DatasetMeta.create({
      name: body.name.trim(),
      source: body.source.trim(),
      filename: body.filename.trim(),
      fileType: body.fileType,
      filePath: body.filePath || "", // jika tidak ada path bisa dikosongkan atau di-generate otomatis
      status: "raw", // default sementara
      description: body.description || "",
      collectionTarget: body.collectionTarget.trim(),
      month: body.month,
      timestamp: new Date(body.timestamp),
    });

    return c.json(
      {
        message: "Dataset metadata uploaded successfully",
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
