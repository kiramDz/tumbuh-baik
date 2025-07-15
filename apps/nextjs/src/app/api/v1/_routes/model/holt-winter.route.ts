import { Hono } from "hono";
import db from "@/lib/database/db";
import { HoltWinterDaily, HoltWinterSummary } from "@/lib/database/schema/model/holt-winter.model";
import { parseError } from "@/lib/utils";

const holtWinter = new Hono();

holtWinter.get("/daily", async (c) => {
  try {
    await db();
    const page = Number(c.req.query("page")) || 1;
    const pageSize = Number(c.req.query("pageSize")) || 10;
    const totalData = await HoltWinterDaily.countDocuments();
    // Tambahkan ini di route untuk debug
    console.log("Collection name:", HoltWinterDaily.collection.name);
    console.log("🟡 Total documents:", totalData);

    const data = await HoltWinterDaily.find()
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .sort({ forecast_date: -1 })
      .lean();

    console.log("🟢 Retrieved documents:", data.length);

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

holtWinter.get("/summary", async (c) => {
  try {
    await db();
    const page = Number(c.req.query("page")) || 1;
    const pageSize = Number(c.req.query("pageSize")) || 10;
    const totalData = await HoltWinterSummary.countDocuments();

    console.log("Collection name:", HoltWinterSummary.collection.name);
    console.log("🟡 Total documents:", totalData);

    const data = await HoltWinterSummary.find()
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .sort({ month: -1 }) // sort by newest month
      .lean();

    console.log("🟢 Retrieved summary documents:", data.length);

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

export default holtWinter;
