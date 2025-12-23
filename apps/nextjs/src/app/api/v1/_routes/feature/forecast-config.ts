import { Hono } from "hono";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";
import { ForecastConfig } from "@/lib/database/schema/feature/forecast-config.model";

const forecastConfigRoute = new Hono();

forecastConfigRoute.post("/", async (c) => {
  try {
    await db();
    const { name, columns, startDate } = await c.req.json();
    const { name, columns, startDate } = await c.req.json();

    if (!name || !Array.isArray(columns) || columns.length === 0) {
      return c.json({ message: "Name and columns are required" }, 400);
    }

    if (!startDate) {
      return c.json({ message: "Start date is required" }, 400);
    }

    if (!startDate) {
      return c.json({ message: "Start date is required" }, 400);
    }

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

    const start = new Date(startDate);
    const end = new Date(start);
    end.setFullYear(end.getFullYear() + 1);

    const start = new Date(startDate);
    const end = new Date(start);
    end.setFullYear(end.getFullYear() + 1);

    const doc = {
      name: name.trim(),
      columns,
      startDate: start,
      endDate: end,
      columns,
      startDate: start,
      endDate: end,
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

forecastConfigRoute.get("/", async (c) => {
  try {
    await db();
    const data = await ForecastConfig.find().sort({ createdAt: -1 });
    return c.json({ data });
  } catch (error) {
    console.error("Error fetching forecast configs:", error);
    return c.json({ message: "Failed to fetch configs" }, 500);
  }
});

export default forecastConfigRoute;