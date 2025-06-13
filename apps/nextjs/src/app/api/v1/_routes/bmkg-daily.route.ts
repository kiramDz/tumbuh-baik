
import { Hono } from "hono";
import db from "@/lib/database/db";
import { DailySummary } from "@/lib/database/schema/plantSummary.model";
import { parseError } from "@/lib/utils";

const bmkgDailyRoute = new Hono();

bmkgDailyRoute.get("/all", async (c) => {
  try {
    await db();
    const dailyData = await DailySummary.find().sort({ forecast_date: 1 }); // Urutkan berdasarkan tanggal

    return c.json({
      message: "Success",
      data: dailyData,
    });
  } catch (error) {
    const err = parseError(error);
    return c.json({ message: err.message }, 500);
  }
});

export default bmkgDailyRoute;