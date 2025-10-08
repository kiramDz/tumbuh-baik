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
      Model = (mongoose.models[slug] ||
        mongoose.model(
          slug,
          new mongoose.Schema({}, { strict: false }),
          slug
        )) as mongoose.Model<any>;
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

datasetMetaRoute.get("/rainfall-summary", async (c) => {
  try {
    await db();
    const Model =
      mongoose.models["rainfall"] ||
      mongoose.model(
        "rainfall",
        new mongoose.Schema({}, { strict: false }),
        "rainfall"
      );

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

// PUT - Update dataset meta by ID or collectionName
datasetMetaRoute.put("/:idOrSlug", async (c) => {
  try {
    await db();

    const { idOrSlug } = c.req.param();

    // Definisikan tipe data untuk body
    let body: Record<string, any> = {}; // atau bisa menggunakan interface yang lebih spesifik

    try {
      const contentType = c.req.header("content-type") || "";

      if (contentType.includes("application/json")) {
        body = (await c.req.json()) as Record<string, any>;
      } else {
        // Gunakan query parameters
        const queries = c.req.queries();
        Object.keys(queries).forEach((key) => {
          body[key] = queries[key][0]; // Ambil nilai pertama jika array
        });
      }
    } catch (parseError) {
      // Fallback ke query parameters jika parsing JSON gagal
      const queries = c.req.queries();
      Object.keys(queries).forEach((key) => {
        body[key] = queries[key][0];
      });
    }

    // Validasi bahwa ada data untuk di-update
    if (Object.keys(body).length === 0) {
      return c.json(
        {
          message:
            "No data provided for update. Send data as JSON body or query parameters.",
        },
        400
      );
    }

    let updatedDataset;

    // Check if idOrSlug is a valid MongoDB ObjectId
    if (mongoose.Types.ObjectId.isValid(idOrSlug)) {
      // Update by ID
      updatedDataset = await DatasetMeta.findByIdAndUpdate(
        idOrSlug,
        { $set: body },
        { new: true, runValidators: true, lean: true }
      );
    } else {
      // Update by collectionName (slug)
      updatedDataset = await DatasetMeta.findOneAndUpdate(
        { collectionName: idOrSlug },
        { $set: body },
        { new: true, runValidators: true, lean: true }
      );
    }

    if (!updatedDataset) {
      return c.json({ message: "Dataset not found" }, 404);
    }

    return c.json(
      {
        message: "Dataset metadata updated successfully",
        data: updatedDataset,
      },
      200
    );
  } catch (error) {
    console.error("Update dataset error:", error);
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
    const parsedData = data.map((item) => ({
      ...item,
      Date: item.Date ? new Date(item.Date) : null, // konversi ke tipe Date
    }));
    const dynamicModel = mongoose.model(
      collectionName,
      new mongoose.Schema({}, { strict: false }),
      collectionName
    );
    await dynamicModel.insertMany(parsedData);

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
      isAPI: false,
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
      isAPI: false,
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

// DELETE - Hapus dataset (metadata + collection)
datasetMetaRoute.delete("/:collectionName", async (c) => {
  try {
    await db();
    const { collectionName } = c.req.param();

    // 1. Hapus metadata
    await DatasetMeta.deleteOne({ collectionName });

    // 2. Drop koleksi MongoDB menggunakan model
    try {
      // Cek apakah model sudah ada
      let Model;
      if (mongoose.models[collectionName]) {
        Model = mongoose.models[collectionName];
      } else {
        Model = mongoose.model(
          collectionName,
          new mongoose.Schema({}, { strict: false }),
          collectionName
        );
      }

      // Drop collection menggunakan model
      await Model.collection.drop();
    } catch (dropError: any) {
      // Jika collection tidak ada, abaikan error
      if (dropError.code !== 26) {
        // 26 = NamespaceNotFound
        throw dropError;
      }
    }

    return c.json({ message: "Dataset deleted successfully" }, 200);
  } catch (error) {
    console.error("Delete dataset error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});
// datasetMetaRoute.delete("/:collectionName", async (c) => {
//   try {
//     await db();
//     const { collectionName } = c.req.param();

//     // 1. Hapus metadata
//     await DatasetMeta.deleteOne({ collectionName });

//     // 2. Drop koleksi MongoDB
//     const connection = mongoose.connection;
//     const collections = await connection.db.listCollections().toArray();
//     const exists = collections.some((col) => col.name === collectionName);

//     if (exists) {
//       await connection.db.dropCollection(collectionName);
//     }

//     return c.json({ message: "Dataset deleted successfully" }, 200);
//   } catch (error) {
//     console.error("Delete dataset error:", error);
//     const { message, status } = parseError(error);
//     return c.json({ message }, status);
//   }
// });

export default datasetMetaRoute;
