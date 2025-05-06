import { Hono } from "hono";
import { BMKGApi } from "@/lib/database/schema/bmkgApi.model";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";


const bmkgApiRoute = new Hono();

bmkgApiRoute.get("/all", async (c) => {
  try {
    await db();
    const allData = await BMKGApi.find().sort({ createdAt: -1 }).lean();
    return c.json({
      message: "Success",
      description: "",
      data: allData,
    });
  } catch (error) {
    const err = parseError(error);
    return c.json({ message: "Error", description: err }, { status: 500 });
  }
});

export default bmkgApiRoute;
