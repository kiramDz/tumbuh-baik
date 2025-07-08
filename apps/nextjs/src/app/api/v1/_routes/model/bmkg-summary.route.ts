import { Hono } from "hono";
import db from "@/lib/database/db";
import { PlantSummary } from "@/lib/database/schema/model/plantSummary.model";
import { parseError } from "@/lib/utils";

const bmkgSummaryRoute = new Hono();

bmkgSummaryRoute.get("/all", async (c) => {
  try {
    await db();
    const summaryData = await PlantSummary.find().sort({ month: 1 }); // sort agar urut

    return c.json({
      message: "Success",
      data: summaryData,
    });
  } catch (error) {
    const err = parseError(error);
    return c.json({ message: "Error", description: err }, { status: 500 });
  }
});

export default bmkgSummaryRoute;
