import { Hono } from "hono";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";
import { LSTMConfig } from "@/lib/database/schema/feature/lstm-config.model";

const lstmConfigRoute = new Hono();

lstmConfigRoute.post("/", async (c) => {
    try {
        await db();
        const { name, columns } = await c.req.json();

        if (!name || !Array.isArray(columns) || columns.length === 0) {
            return c.json({ message: "Name and columns are required" }, 400);
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
    
    const doc = {
        name: name.trim(),
        columns,
        status: "pending",
        forecastResultCollection: `lstm_forecast_${name.toLowerCase().replace(/\s+/g, "_")}_${Date.now()}`,
    };

    const result = await LSTMConfig.create(doc);

    return c.json({ message: "Config saved", data: result });
} catch (error) {
    console.error("LSTM config error:", error);
    const { message, status } = parseError(error);
    return c.json({ message }, status);
}
});

lstmConfigRoute.get("/", async (c) => {
    try {
        await db();
        const data = await LSTMConfig.find().sort({ createdAt: -1 });
        return c.json({ data });
        
    } catch (error) {
      console.error("Error fetching LSTM configs:", error);
      return c.json({ message: "Failed to fetch configs" }, 500);
    }
});

export default lstmConfigRoute;
