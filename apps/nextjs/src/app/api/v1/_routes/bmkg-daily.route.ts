import { Hono } from "hono";
import db from "@/lib/database/db";
import { DailySummary } from "@/lib/database/schema/plantSummary.model";
import { parseError } from "@/lib/utils";

const bmkgDailyRoute = new Hono();

bmkgDailyRoute.get("/", async (c) => {
  try {
    await db();
    const page = Number(c.req.query("page")) || 1;
    const pageSize = Number(c.req.query("pageSize")) || 10;
    const totalData = await DailySummary.countDocuments();

    const data = await DailySummary.find()
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .sort({ forecast_date: -1 }) 
      .lean();

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
    const err = parseError(error);
    return c.json({ message: err.message }, 500);
  }
});

export default bmkgDailyRoute;
