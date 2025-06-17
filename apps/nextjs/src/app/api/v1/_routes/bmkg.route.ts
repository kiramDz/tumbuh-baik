import { Hono } from "hono";
import db from "@/lib/database/db";
import { BmkgData } from "@/lib/database/schema/bmkg.model";
import { parseError } from "@/lib/utils";

const bmkgRoute = new Hono();

bmkgRoute.get("/", async (c) => {
  try {
    console.log("=== Endpoint BMKG Data terpanggil ===");
    await db();

    const page = Number(c.req.query("page")) || 1;
    // const PAGE_SIZE = 10;
    const pageSize = Number(c.req.query("pageSize") || "10");

    const totalData = await BmkgData.countDocuments();

    const data = await BmkgData.find()
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .sort({ Date: -1 }) // urut terbaru dulu
      .lean();

    return c.json(
      {
        message: "Success",
        description: "Fetched BMKG data",
        data: {
          items: data,
          total: totalData,
          currentPage: page,
          totalPages: Math.ceil(totalData / pageSize),
          pageSize
        },
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error fetching BMKG data:", error);
    return c.json(
      {
        message: "Error",
        description: parseError(error),
        data: null,
      },
      { status: 500 }
    );
  }
});

export default bmkgRoute;
