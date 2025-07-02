import { Hono } from "hono";
import db from "@/lib/database/db";
import { BuoysData } from "@/lib/database/schema/dataset/buoys.model";
import { parseError } from "@/lib/utils";

const buoysRoute = new Hono();

buoysRoute.get("/", async (c) => {
  try {
    console.log("=== Endpoint Buoys Data terpanggil ===");
    await db();
    const page = Number(c.req.query("page")) || 1;
    const pageSize = Number(c.req.query("pageSize") || "10");

    const sortBy = c.req.query("sortBy") || "Date";
    const sortOrder = c.req.query("sortOrder") || "desc";

    const totalData = await BuoysData.countDocuments();

    const sortQuery: Record<string, 1 | -1> = {
      [sortBy]: sortOrder === "desc" ? -1 : 1,
    };
    const data = await BuoysData.find()
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .sort(sortQuery) // Dynamic sorting
      .lean();

    return c.json(
      {
        message: "Success",
        description: "Fetched Buoy data",
        data: {
          items: data,
          total: totalData,
          currentPage: page,
          totalPages: Math.ceil(totalData / pageSize),
          pageSize,
          sortBy,
          sortOrder, // Return sorting info
        },
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error fetching Buoy data:", error);
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

export default buoysRoute;
