import { Hono } from "hono";
import mongoose, { mongo } from "mongoose";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";

const preprocessingRoute = new Hono();

// GET - Fetch preprocessing report by id
preprocessingRoute.get("/:id", async (c) => {
  try {
    const id = c.req.param("id");
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return c.json({ message: "Invalid preprocessing ID format" }, 400);
    }
    await db();

    // Using raw db connection since collection has dynamic mixed schema with Hard Guard - TypeScript safety
    const mongoDb = mongoose.connection.db;
    if (!mongoDb) {
      throw new Error("MongoDB connection not initialized");
    }

    const report = await mongoDb
      .collection("preprocessing_report")
      .findOne({ _id: new mongoose.Types.ObjectId(id) });

    if (!report) {
      return c.json({ message: "Preprocessing report not found" }, 404);
    }
    return c.json({ data: report }, 200);
  } catch (error) {
    console.error("Fetch preprocessing report error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

// GET - Fetch Decomposition Result by Preprocessing ID
preprocessingRoute.get("/:id/decomposition", async (c) => {
  try {
    const id = c.req.param("id");
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return c.json({ message: "Invalid preprocessing ID format" }, 400);
    }

    await db();

    // Fetch the linked decomposition report with Hard Guard - TypeScript safety
    const mongoDb = mongoose.connection.db;
    if (!mongoDb) {
      throw new Error("MongoDB connection not initialized");
    }

    const decomposition = await mongoDb
      .collection("decomposition_report")
      .findOne({ preprocessing_id: new mongoose.Types.ObjectId(id) });

    if (!decomposition) {
      return c.json(
        { message: "Decomposition data not found for this preprocessing ID" },
        404,
      );
    }

    return c.json({ data: decomposition }, 200);
  } catch (error) {
    console.error("Fetch decomposition error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});

export default preprocessingRoute;
