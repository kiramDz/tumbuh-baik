import { Hono } from "hono";
import db from "@/lib/database/db";
import { DatasetMeta } from "@/lib/database/schema/feature/dataset-meta.model";
import { parseError } from "@/lib/utils";
import mongoose from "mongoose";

const historicalLstmRoute = new Hono();

/**
 * GET /historical-lstm/:collectionName/:columnName
 * Fetch all historical data from a specific collection and column
 * Used for combining historical data with LSTM forecast
 */
historicalLstmRoute.get("/:collectionName/:columnName", async (c) => {
  try {
    await db();
    const { collectionName, columnName } = c.req.param();

    console.log(`[Historical LSTM] Fetching data from ${collectionName}, column: ${columnName}`);

    // Verify collection exists in metadata
    const meta = await DatasetMeta.findOne({ collectionName }).lean();
    if (!meta) {
      return c.json({ 
        message: `Dataset with collection name "${collectionName}" not found` 
      }, 404);
    }

    // Verify column exists
    const columns = (meta as any).columns || [];
    if (!columns.includes(columnName)) {
      return c.json({ 
        message: `Column "${columnName}" not found in dataset "${collectionName}". Available columns: ${columns.join(", ")}` 
      }, 404);
    }

    // Get or create mongoose model for the collection
    let Model;
    try {
      Model = mongoose.model(collectionName);
    } catch {
      Model = mongoose.models[collectionName] || 
              mongoose.model(collectionName, new mongoose.Schema({}, { strict: false }), collectionName);
    }

    // Fetch all data sorted by Date ascending
    const data = await Model.find({
      Date: { $exists: true },
      [columnName]: { $exists: true, $ne: null }
    })
      .sort({ Date: 1 })
      .select(`Date ${columnName}`)
      .lean();

    console.log(`[Historical LSTM] Found ${data.length} records`);

    // Transform to simplified format
    const historicalData = data.map((item: any) => ({
      date: item.Date,
      value: item[columnName]
    }));

    return c.json({
      message: "Success",
      data: {
        collectionName,
        columnName,
        total: historicalData.length,
        items: historicalData
      }
    });

  } catch (error) {
    console.error("[Historical LSTM] Error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

/**
 * POST /historical-lstm/batch
 * Fetch multiple historical datasets in one request
 * Body: { datasets: [{ collectionName, columnName }] }
 */
historicalLstmRoute.post("/batch", async (c) => {
  try {
    await db();
    const body = await c.req.json();
    const { datasets } = body;

    if (!Array.isArray(datasets) || datasets.length === 0) {
      return c.json({ 
        message: "datasets must be a non-empty array of { collectionName, columnName }" 
      }, 400);
    }

    console.log(`[Historical LSTM Batch] Fetching ${datasets.length} datasets`);

    const results: Record<string, any> = {};
    const errors: string[] = [];

    for (const { collectionName, columnName } of datasets) {
      try {
        // Verify collection exists
        const meta = await DatasetMeta.findOne({ collectionName }).lean();
        if (!meta) {
          errors.push(`Collection "${collectionName}" not found`);
          continue;
        }

        // Verify column exists
        const columns = (meta as any).columns || [];
        if (!columns.includes(columnName)) {
          errors.push(`Column "${columnName}" not found in "${collectionName}"`);
          continue;
        }

        // Get or create model
        let Model;
        try {
          Model = mongoose.model(collectionName);
        } catch {
          Model = mongoose.models[collectionName] || 
                  mongoose.model(collectionName, new mongoose.Schema({}, { strict: false }), collectionName);
        }

        // Fetch data
        const data = await Model.find({
          Date: { $exists: true },
          [columnName]: { $exists: true, $ne: null }
        })
          .sort({ Date: 1 })
          .select(`Date ${columnName}`)
          .lean();

        // Store with composite key
        const key = `${collectionName}::${columnName}`;
        results[key] = data.map((item: any) => ({
          date: item.Date,
          value: item[columnName]
        }));

        console.log(`[Historical LSTM Batch] ${key}: ${results[key].length} records`);

      } catch (error) {
        console.error(`[Historical LSTM Batch] Error processing ${collectionName}:${columnName}:`, error);
        errors.push(`Error processing ${collectionName}:${columnName}`);
      }
    }

    return c.json({
      message: errors.length > 0 ? "Partial success" : "Success",
      data: results,
      errors: errors.length > 0 ? errors : undefined,
      stats: {
        requested: datasets.length,
        succeeded: Object.keys(results).length,
        failed: errors.length
      }
    });

  } catch (error) {
    console.error("[Historical LSTM Batch] Error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

export default historicalLstmRoute;