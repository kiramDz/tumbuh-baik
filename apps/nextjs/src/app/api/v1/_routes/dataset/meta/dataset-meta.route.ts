import { Hono } from "hono";
import db from "@/lib/database/db";
import { DatasetMeta } from "@/lib/database/schema/feature/dataset-meta.model";
import { parseError } from "@/lib/utils";
import mongoose from "mongoose";

interface ConversionInfo {
  originalFormat: string;
  convertedTo: string;
  isMultiFile: boolean;
  filesProcessed: number;
  totalRecords: number;
  fileSize: number;
  processingDetails?: {
    filesSuccessful: number;
    filesFailed: number;
    totalFiles: number;
    duplicatesRemoved: number;
    dateRange: {
      start: string | null;
      end: string | null;
      years?: number;
    };
    failedFiles: string[];
    warnings: string[];
  };
}

interface UploadResponse {
  message: string;
  data: any;
  conversionInfo: ConversionInfo;
}

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

    const total = await DatasetMeta.countDocuments({
      deletedAt: { $ne: null },
    });
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

// Route untuk fetch semua data tanpa pagination (khusus chart)
datasetMetaRoute.get("/:slug/chart-data", async (c) => {
  try {
    await db();
    const { slug } = c.req.param();
    console.log("[DEBUG] API chart-data called with slug:", slug);

    // Gunakan type assertion untuk meta
    const meta = (await DatasetMeta.findOne({
      collectionName: slug,
    }).lean()) as {
      name: string;
      collectionName: string;
      columns: string[];
      [key: string]: any;
    } | null;

    if (!meta) return c.json({ message: "Dataset not found" }, 404);

    // Validasi kolom Date
    const hasDateColumn = meta.columns.some(
      (col: string) => col.toLowerCase() === "date"
    );

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
      Model = (mongoose.models[slug] ||
        mongoose.model(
          slug,
          new mongoose.Schema({}, { strict: false }),
          slug
        )) as mongoose.Model<any>;
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

    const contentType = c.req.header("content-type") || "";

    // Handle XLSX via multipart/form-data for single upload and multi-file upload
    if (contentType.includes("multipart/form-data")) {
      return await handleXlsxUpload(c);
    }
    // ✅ Handle CSV/JSON via JSON body (existing logic)
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
      data,
      filename = `${name}.${fileType}`,
      description = "",
      status = "raw",
      collectionName: rawCollectionName,
    } = body;

    if (!Array.isArray(data) || data.length === 0) {
      return c.json({ message: "data must be a non-empty array" }, 400);
    }

    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB
    const collectionName =
      rawCollectionName?.trim() ||
      name
        .trim()
        .replace(/[^a-zA-Z0-9\s]/g, "") // Keep spaces, remove special chars
        .replace(/\s+/g, " "); // Normalize spaces
    const fileSize = Buffer.byteLength(JSON.stringify(data));

    if (fileSize > MAX_FILE_SIZE) {
      return c.json({ message: "File size exceeds 16MB limit" }, 400);
    }

    const totalRecords = data.length || 0;
    const columns = data[0] ? Object.keys(data[0]) : [];

    // Insert data ke collection dinamis
    const parsedData = data.map((item) => ({
      ...item,
      Date: item.Date ? new Date(item.Date) : null,
    }));

    // Check if model already exists, if not create it
    let dynamicModel;
    try {
      dynamicModel = mongoose.model(collectionName);
    } catch {
      dynamicModel = mongoose.model(
        collectionName,
        new mongoose.Schema({}, { strict: false }),
        collectionName
      );
    }

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
async function handleXlsxUpload(c: any) {
  try {
    const formData = await c.req.formData();

    // Extract files from formData
    const name = (formData.get("name") as string)?.trim();
    const source = (formData.get("source") as string)?.trim();
    const description = ((formData.get("description") as string) || "").trim();
    const status = ((formData.get("status") as string) || "raw").trim();
    const isMultiFile = formData.get("isMultiFile") === "true";

    // Basic validation
    if (!name || !source) {
      return c.json({ message: "name and source are required" }, 400);
    }

    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB
    const MAX_TOTAL_SIZE = 100 * 1024 * 1024; // 100 MB for multi-file
    const MAX_FILES = 50; // Maximum 50 files

    let conversionResult;
    let filename: string;
    let filesProcessed = 0;

    // ✅ Handle Multi-file XLSX merge
    if (isMultiFile) {
      console.log("Processing multi-file XLSX upload...");

      // Collect all files from formData
      const files: Array<{ buffer: number[]; filename: string }> = [];
      let totalSize = 0;

      // Get all file entries (should be named "files")
      const fileEntries = formData.getAll("files");

      if (!fileEntries || fileEntries.length === 0) {
        return c.json(
          { message: "No files provided for multi-file upload" },
          400
        );
      }

      if (fileEntries.length > MAX_FILES) {
        return c.json(
          {
            message: `Too many files. Maximum ${MAX_FILES} files allowed per batch`,
          },
          400
        );
      }

      // Process each file
      for (const entry of fileEntries) {
        if (entry instanceof File) {
          const file = entry as File;

          // Validate file extension
          if (!file.name.toLowerCase().endsWith(".xlsx")) {
            return c.json(
              {
                message: `File ${file.name} is not a .xlsx file`,
              },
              400
            );
          }

          // Validate individual file size
          if (file.size > MAX_FILE_SIZE) {
            return c.json(
              {
                message: `File ${file.name} exceeds 16MB limit`,
              },
              400
            );
          }

          totalSize += file.size;

          // Validate total size
          if (totalSize > MAX_TOTAL_SIZE) {
            return c.json(
              {
                message: `Total file size exceeds 100MB limit`,
              },
              400
            );
          }

          const arrayBuffer = await file.arrayBuffer();
          const fileBuffer = Array.from(new Uint8Array(arrayBuffer));

          files.push({
            buffer: fileBuffer,
            filename: file.name,
          });
        }
      }

      if (files.length === 0) {
        return c.json(
          {
            message: "No valid .xlsx files found",
          },
          400
        );
      }

      filesProcessed = files.length;
      filename = `${name.replace(/\s+/g, "_")}_merged_${
        files.length
      }_files.csv`;

      // ✅ Send to Flask for merging
      const flaskFormData = new FormData();

      // Add each file to FormData for Flask
      files.forEach((fileData) => {
        const buffer = Buffer.from(fileData.buffer);
        const flaskFile = new File([buffer], fileData.filename, {
          type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        });
        flaskFormData.append("files", flaskFile);
      });

      try {
        console.log(`Sending ${files.length} files to Flask merge service...`);
        const response = await fetch(
          "http://localhost:5001/api/v1/convert/xlsx-merge-csv",
          {
            method: "POST",
            body: flaskFormData,
          }
        );

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(
            `Flask merge service error: ${response.status} ${response.statusText}. Response: ${errorText}`
          );
        }

        const responseText = await response.text();
        try {
          conversionResult = JSON.parse(responseText);
        } catch (jsonError) {
          console.error("JSON Parse Error:", jsonError);
          console.error(
            "Raw Response:",
            responseText.substring(0, 500) + "..."
          );
          throw new Error(
            `Invalid JSON response from merge service: ${jsonError}`
          );
        }
      } catch (fetchError) {
        console.error("Flask merge service error:", fetchError);
        return c.json(
          { message: `Failed to connect to merge service: ${fetchError}` },
          500
        );
      }
    }
    // ✅ Handle Single XLSX file
    else {
      console.log("Processing single XLSX file upload...");

      const file = formData.get("file") as File;

      if (!file || !(file instanceof File)) {
        return c.json(
          { message: "file is required for single file upload" },
          400
        );
      }

      // Validate file extension
      if (!file.name.toLowerCase().endsWith(".xlsx")) {
        return c.json({ message: "Only .xlsx files are allowed" }, 400);
      }

      // Validate file size
      if (file.size > MAX_FILE_SIZE) {
        return c.json({ message: "File size exceeds 16MB limit" }, 400);
      }

      filesProcessed = 1;
      filename = file.name.replace(".xlsx", ".csv");

      // Convert File to buffer array
      const arrayBuffer = await file.arrayBuffer();
      const fileBuffer = Array.from(new Uint8Array(arrayBuffer));

      // ✅ Send to Flask for single file conversion
      const flaskFormData = new FormData();
      const buffer = Buffer.from(fileBuffer);
      const flaskFile = new File([buffer], file.name, {
        type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      });
      flaskFormData.append("file", flaskFile);

      try {
        console.log(`Sending single file to Flask conversion service...`);
        const response = await fetch(
          "http://localhost:5001/api/v1/convert/xlsx-to-csv",
          {
            method: "POST",
            body: flaskFormData,
          }
        );

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(
            `Flask service error: ${response.status} ${response.statusText}. Response: ${errorText}`
          );
        }

        const responseText = await response.text();
        try {
          conversionResult = JSON.parse(responseText);
        } catch (jsonError) {
          console.error("JSON Parse Error:", jsonError);
          console.error(
            "Raw Response:",
            responseText.substring(0, 500) + "..."
          );
          throw new Error(
            `Invalid JSON response from conversion service: ${jsonError}`
          );
        }
      } catch (fetchError) {
        console.error("Flask service error:", fetchError);
        return c.json(
          { message: `Failed to connect to conversion service: ${fetchError}` },
          500
        );
      }
    }

    // ✅ Check conversion result (same for both single and multi)
    if (conversionResult.status !== "success") {
      return c.json(
        {
          message: `XLSX conversion failed: ${
            conversionResult.error || "Unknown error"
          }`,
        },
        400
      );
    }

    // Extract converted data
    const processedData = conversionResult.records;
    const totalRecords = conversionResult.record_count;
    const columns = conversionResult.columns;

    if (!Array.isArray(processedData) || processedData.length === 0) {
      return c.json({ message: "Conversion resulted in empty data" }, 400);
    }

    // Generate collection name
    const collectionName = name
      .trim()
      .replace(/[^a-zA-Z0-9\s]/g, "")
      .replace(/\s+/g, " ");

    const fileSize = Buffer.byteLength(JSON.stringify(processedData));

    // Validate processed data size
    if (fileSize > MAX_TOTAL_SIZE) {
      return c.json({ message: "Processed data exceeds size limit" }, 400);
    }

    // Parse data with Date conversion
    const parsedData = processedData.map((item: any) => ({
      ...item,
      Date: item.Date ? new Date(item.Date) : null,
    }));

    // Create or get dynamic model
    let dynamicModel;
    try {
      dynamicModel = mongoose.model(collectionName);
    } catch {
      dynamicModel = mongoose.model(
        collectionName,
        new mongoose.Schema({}, { strict: false }),
        collectionName
      );
    }

    // Insert to MongoDB
    await dynamicModel.insertMany(parsedData);

    // Save metadata
    const newDataset = await DatasetMeta.create({
      name: name.trim(),
      source: source.trim(),
      filename,
      collectionName,
      fileType: "csv", // Always saved as CSV after conversion
      status: status.trim(),
      description: description.trim(),
      fileSize,
      totalRecords,
      columns,
      isAPI: false,
    });

    // ✅ Return response with interface UploadResponse
    const response: UploadResponse = {
      message: "Dataset uploaded and metadata saved successfully",
      data: newDataset,
      conversionInfo: {
        originalFormat: "xlsx",
        convertedTo: "csv",
        isMultiFile,
        filesProcessed,
        totalRecords,
        fileSize,
      },
    };

    // Add additional info for multi-file uploads
    if (isMultiFile && conversionResult.processing_summary) {
      response.conversionInfo.processingDetails = {
        filesSuccessful: conversionResult.processing_summary.files_processed,
        filesFailed: conversionResult.processing_summary.files_failed,
        totalFiles: conversionResult.processing_summary.total_files,
        duplicatesRemoved:
          conversionResult.processing_summary.duplicates_removed,
        dateRange: conversionResult.processing_summary.date_range,
        failedFiles: conversionResult.failed_files || [],
        warnings: conversionResult.warnings || [],
      };
    }

    console.log(
      `✅ XLSX conversion completed: ${
        isMultiFile ? "Multi-file" : "Single"
      } upload successful`
    );
    return c.json(response, 201);
  } catch (error) {
    console.error("XLSX upload error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
}

// datasetMetaRoute.post("/", async (c) => {
//   try {
//     await db();
//     const body = await c.req.json();

//     const requiredFields = ["name", "source", "fileType", "data"];
//     for (const field of requiredFields) {
//       if (!body[field]) {
//         return c.json({ message: `${field} is required` }, 400);
//       }
//     }

//     if (!["csv", "json"].includes(body.fileType)) {
//       return c.json({ message: "fileType must be 'csv' or 'json'" }, 400);
//     }

//     const {
//       name,
//       source,
//       fileType,
//       data, // data records (parsed JSON/CSV)
//       filename = `${name}.${fileType}`,
//       description = "",
//       status = "raw",
//       collectionName: rawCollectionName,
//     } = body;

//     if (!Array.isArray(data) || data.length === 0) {
//       return c.json({ message: "data must be a non-empty array" }, 400);
//     }

//     const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB
//     const collectionName = rawCollectionName?.trim() || name.trim();
//     const fileSize = Buffer.byteLength(JSON.stringify(body.data));

//     if (fileSize > MAX_FILE_SIZE) {
//       return c.json({ message: "File size exceeds 16MB limit" }, 400);
//     }

//     const totalRecords = data.length || 0;
//     const columns = data[0] ? Object.keys(data[0]) : [];

//     // Insert data ke collection dinamis
//     const parsedData = data.map((item) => ({
//       ...item,
//       Date: item.Date ? new Date(item.Date) : null, // konversi ke tipe Date
//     }));

//     // Check if model already exists, if not create it
//     let dynamicModel;
//     try {
//       dynamicModel = mongoose.model(collectionName);
//     } catch {
//       dynamicModel = mongoose.model(
//         collectionName,
//         new mongoose.Schema({}, { strict: false }),
//         collectionName
//       );
//     }

//     await dynamicModel.insertMany(parsedData);

//     // Simpan metadata
//     const newDataset = await DatasetMeta.create({
//       name: name.trim(),
//       source: source.trim(),
//       filename,
//       collectionName,
//       fileType,
//       status,
//       description,
//       fileSize,
//       totalRecords,
//       columns,
//       isAPI: false,
//     });

//     console.log({
//       name,
//       source,
//       filename,
//       collectionName,
//       fileType,
//       status,
//       description,
//       fileSize,
//       totalRecords,
//       columns,
//       isAPI: false,
//     });

//     return c.json(
//       {
//         message: "Dataset uploaded and metadata saved successfully",
//         data: newDataset,
//       },
//       201
//     );
//   } catch (error) {
//     console.error("Upload dataset error:", error);
//     const { message, status } = parseError(error);
//     return c.json({ message }, status);
//   }
// });
// datasetMetaRoute.post("/", async (c) => {
//   try {
//     await db();
//     const body = await c.req.json();

//     const requiredFields = ["name", "source", "fileType", "data"];
//     for (const field of requiredFields) {
//       if (!body[field]) {
//         return c.json({ message: `${field} is required` }, 400);
//       }
//     }

//     if (!["csv", "json"].includes(body.fileType)) {
//       return c.json({ message: "fileType must be 'csv' or 'json'" }, 400);
//     }

//     const {
//       name,
//       source,
//       fileType,
//       data, // data records (parsed JSON/CSV)
//       filename = `${name}.${fileType}`,
//       description = "",
//       status = "raw",
//       collectionName: rawCollectionName,
//     } = body;

//     if (!Array.isArray(data) || data.length === 0) {
//       return c.json({ message: "data must be a non-empty array" }, 400);
//     }

//     const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB
//     const collectionName = rawCollectionName?.trim() || name.trim();
//     const fileSize = Buffer.byteLength(JSON.stringify(body.data));

//     if (fileSize > MAX_FILE_SIZE) {
//       return c.json({ message: "File size exceeds 16MB limit" }, 400);
//     }

//     const totalRecords = data.length || 0;
//     const columns = data[0] ? Object.keys(data[0]) : [];

//     // Insert data ke collection dinamis
//     const parsedData = data.map((item) => ({
//       ...item,
//       Date: item.Date ? new Date(item.Date) : null, // konversi ke tipe Date
//     }));
//     const dynamicModel = mongoose.model(
//       collectionName,
//       new mongoose.Schema({}, { strict: false }),
//       collectionName
//     );
//     await dynamicModel.insertMany(parsedData);

//     // Simpan metadata
//     const newDataset = await DatasetMeta.create({
//       name: name.trim(),
//       source: source.trim(),
//       filename,
//       collectionName,
//       fileType,
//       status,
//       description,
//       fileSize,
//       totalRecords,
//       columns,
//       isAPI: false,
//     });

//     console.log({
//       name,
//       source,
//       filename,
//       collectionName,
//       fileType,
//       status,
//       description,
//       fileSize,
//       totalRecords,
//       columns,
//       isAPI: false,
//     });

//     return c.json(
//       {
//         message: "Dataset uploaded and metadata saved successfully",
//         data: newDataset,
//       },
//       201
//     );
//   } catch (error) {
//     console.error("Upload dataset error:", error);
//     const { message, status } = parseError(error);
//     return c.json({ message }, status);
//   }
// });

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
