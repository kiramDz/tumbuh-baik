// apps/nextjs/src/app/api/v1/_routes/model/decompose-lstm.route.ts
import { Hono } from "hono";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";
import mongoose from "mongoose";

const decomposeLstmRoute = new Hono();

// GET all decompose data
decomposeLstmRoute.get("/all", async (c) => {
  try {
    await db();
    
    const DecomposeModel = mongoose.models["decompose-lstm"] || 
      mongoose.model("decompose-lstm", new mongoose.Schema({}, { strict: false }), "decompose-lstm");
    
    const data = await DecomposeModel.find()
      .sort({ date: 1 })
      .lean();

    console.log(`📊 Retrieved ${data.length} decompose documents`);

    return c.json({
      message: "Success",
      data: data,
    });
  } catch (error) {
    console.error("Error fetching decompose LSTM data:", error);
    const err = parseError(error);
    return c.json({ message: err.message }, 500);
  }
});

// GET decompose data by date
decomposeLstmRoute.get("/date/:date", async (c) => {
  try {
    await db();
    const { date } = c.req.param();
    
    const DecomposeModel = mongoose.models["decompose-lstm"] || 
      mongoose.model("decompose-lstm", new mongoose.Schema({}, { strict: false }), "decompose-lstm");
    
    const data = await DecomposeModel.findOne({ date }).lean();

    if (!data) {
      return c.json({ message: "Data not found" }, 404);
    }

    return c.json({
      message: "Success",
      data: data,
    });
  } catch (error) {
    console.error("Error fetching decompose by date:", error);
    const err = parseError(error);
    return c.json({ message: err.message }, 500);
  }
});

// GET decompose data by config_id
decomposeLstmRoute.get("/config/:configId", async (c) => {
  try {
    await db();
    const { configId } = c.req.param();
    
    const DecomposeModel = mongoose.models["decompose-lstm"] || 
      mongoose.model("decompose-lstm", new mongoose.Schema({}, { strict: false }), "decompose-lstm");
    
    const data = await DecomposeModel.find({ config_id: configId })
      .sort({ date: 1 })
      .lean();

    console.log(`📊 Retrieved ${data.length} decompose documents for config ${configId}`);

    return c.json({
      message: "Success",
      data: data,
    });
  } catch (error) {
    console.error("Error fetching decompose by config:", error);
    const err = parseError(error);
    return c.json({ message: err.message }, 500);
  }
});

// GET decompose data with pagination
decomposeLstmRoute.get("/paginated", async (c) => {
  try {
    await db();
    const page = Number(c.req.query("page")) || 1;
    const pageSize = Number(c.req.query("pageSize")) || 10;
    const configId = c.req.query("config_id");
    
    const DecomposeModel = mongoose.models["decompose-lstm"] || 
      mongoose.model("decompose-lstm", new mongoose.Schema({}, { strict: false }), "decompose-lstm");
    
    const filter = configId ? { config_id: configId } : {};
    
    const [totalData, data] = await Promise.all([
      DecomposeModel.countDocuments(filter),
      DecomposeModel.find(filter)
        .skip((page - 1) * pageSize)
        .limit(pageSize)
        .sort({ date: 1 })
        .lean(),
    ]);

    return c.json({
      message: "Success",
      data: {
        items: data,
        total: totalData,
        currentPage: page,
        totalPages: Math.ceil(totalData / pageSize),
        pageSize,
      },
    });
  } catch (error) {
    console.error("Error fetching paginated decompose:", error);
    const err = parseError(error);
    return c.json({ message: err.message }, 500);
  }
});

// GET decompose statistics
decomposeLstmRoute.get("/stats", async (c) => {
  try {
    await db();
    const configId = c.req.query("config_id");
    
    const DecomposeModel = mongoose.models["decompose-lstm"] || 
      mongoose.model("decompose-lstm", new mongoose.Schema({}, { strict: false }), "decompose-lstm");
    
    const filter = configId ? { config_id: configId } : {};
    const data = await DecomposeModel.find(filter).lean();

    if (data.length === 0) {
      return c.json({
        message: "No data found",
        data: null,
      });
    }

    // Calculate statistics untuk setiap parameter
    const stats: any = {};
    const parameters = new Set<string>();

    data.forEach((doc: any) => {
      if (doc.parameters) {
        Object.keys(doc.parameters).forEach(param => parameters.add(param));
      }
    });

    parameters.forEach(param => {
      const values = {
        trend: [] as number[],
        seasonal: [] as number[],
        resid: [] as number[],
      };

      data.forEach((doc: any) => {
        if (doc.parameters?.[param]) {
          values.trend.push(doc.parameters[param].trend);
          values.seasonal.push(doc.parameters[param].seasonal);
          values.resid.push(doc.parameters[param].resid);
        }
      });

      stats[param] = {
        trend: {
          min: Math.min(...values.trend),
          max: Math.max(...values.trend),
          mean: values.trend.reduce((a, b) => a + b, 0) / values.trend.length,
        },
        seasonal: {
          min: Math.min(...values.seasonal),
          max: Math.max(...values.seasonal),
          mean: values.seasonal.reduce((a, b) => a + b, 0) / values.seasonal.length,
        },
        resid: {
          min: Math.min(...values.resid),
          max: Math.max(...values.resid),
          mean: values.resid.reduce((a, b) => a + b, 0) / values.resid.length,
        },
        count: values.trend.length,
      };
    });

    return c.json({
      message: "Success",
      data: {
        totalDocuments: data.length,
        dateRange: {
          start: data[0].date,
          end: data[data.length - 1].date,
        },
        parameters: stats,
      },
    });
  } catch (error) {
    console.error("Error calculating decompose stats:", error);
    const err = parseError(error);
    return c.json({ message: err.message }, 500);
  }
});

// DELETE decompose data by config_id
decomposeLstmRoute.delete("/config/:configId", async (c) => {
  try {
    await db();
    const { configId } = c.req.param();
    
    const DecomposeModel = mongoose.models["decompose-lstm"] || 
      mongoose.model("decompose-lstm", new mongoose.Schema({}, { strict: false }), "decompose-lstm");
    
    const result = await DecomposeModel.deleteMany({ config_id: configId });

    return c.json({
      message: "Success",
      data: {
        deletedCount: result.deletedCount,
      },
    });
  } catch (error) {
    console.error("Error deleting decompose data:", error);
    const err = parseError(error);
    return c.json({ message: err.message }, 500);
  }
});

export default decomposeLstmRoute;