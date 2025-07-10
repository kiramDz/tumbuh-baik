import { Hono } from "hono";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";
import { ForecastConfig } from "@/lib/database/schema/feature/forecast-config.model";

const forecastConfigRoute = new Hono();

forecastConfigRoute.post("/", async (c) => {
  try {
    await db();
    const { name, columns } = await c.req.json();

    // Validasi input
    if (!name || !Array.isArray(columns) || columns.length === 0) {
      return c.json({ message: "Name and columns are required" }, 400);
    }

    // Validasi setiap column
    for (const column of columns) {
      if (!column.collectionName || !column.columnName) {
        return c.json(
          {
            message: "Each column must have collectionName and columnName",
          },
          400
        );
      }
    }

    // Buat SATU dokumen dengan array columns
    const doc = {
      name: name.trim(),
      columns, // langsung simpan array columns
      status: "pending",
      forecastResultCollection: `forecast_${name.toLowerCase().replace(/\s+/g, "_")}_${Date.now()}`,
    };

    // Simpan sebagai 1 record
    const result = await ForecastConfig.create(doc);

    return c.json({ message: "Config saved", data: result });
  } catch (error) {
    console.error("Forecast config error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
  }
});



// forecastConfigRoute.post("/", async (c) => {
//   try {
//     await db();
//     const { name, columns } = await c.req.json();

//     if (!name || !Array.isArray(columns) || columns.length === 0) {
//       return c.json({ message: "Invalid request" }, 400);
//     }

//     const docs = columns.map((item) => ({
//       name,
//       collectionName: item.collectionName,
//       columnName: item.columnName,
//       status: "pending",
//       forecastResultCollection: `forecast_${item.collectionName}_${item.columnName}`,
//       createdAt: new Date(),
//     }));

//     await ForecastConfig.insertMany(docs);

//     return c.json({ message: "Config saved", data: docs });
//   } catch (error) {
//     console.error("Forecast config error:", error);
//     const { message, status } = parseError(error);
//     return c.json({ message }, status);
//   }
// });

export default forecastConfigRoute;
