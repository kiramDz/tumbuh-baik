// 1. API Route untuk Export CSV - /app/api/v1/export-csv/route.ts
import { Hono } from "hono";
import db from "@/lib/database/db";
import { BmkgData } from "@/lib/database/schema/dataset/bmkg.model";
import { BuoysData } from "@/lib/database/schema/dataset/buoys.model"; // Sesuaikan path
import { parseError } from "@/lib/utils";
import { convertToCSV } from "@/lib/export";

const exportRoute = new Hono();

// Function untuk convert data ke CSV format
exportRoute.get("/", async (c) => {
  try {
    console.log("=== Export CSV Endpoint terpanggil ===");
    await db();

    const category = c.req.query("category"); // bmkg atau buoys
    const sortBy = c.req.query("sortBy") || "Date";
    const sortOrder = c.req.query("sortOrder") || "desc";

    if (!category || !["bmkg", "buoys"].includes(category)) {
      return c.json(
        {
          message: "Error",
          description: "Invalid category. Must be 'bmkg' or 'buoys'",
          data: null,
        },
        { status: 400 }
      );
    }

    const sortQuery: Record<string, 1 | -1> = {
      [sortBy]: sortOrder === "desc" ? -1 : 1,
    };

    let data: any[] = [];
    let filename = "";
    let columns: string[] = [];

    if (category === "bmkg") {
      data = await BmkgData.find().sort(sortQuery).lean();
      filename = "bmkg_data";
      // Sesuaikan dengan kolom yang ada di bmkgColumns
      columns = ["Date", "Time", "Temperature", "Humidity", "Pressure"]; // Sesuaikan dengan kolom actual
    } else if (category === "buoys") {
      data = await BuoysData.find().sort(sortQuery).lean();
      filename = "buoys_data";
      // Sesuaikan dengan kolom yang ada di buoysColumns
      columns = ["Date", "Time", "WaveHeight", "WavePeriod", "WindSpeed"]; // Sesuaikan dengan kolom actual
    }

    const csvContent = convertToCSV(data, columns);

    // Set headers untuk download file
    c.header("Content-Type", "text/csv;charset=utf-8;");
    c.header("Content-Disposition", `attachment; filename="${filename}_${new Date().toISOString().split("T")[0]}.csv"`);

    return c.text(csvContent);
  } catch (error) {
    console.error("Error exporting CSV:", error);
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

export default exportRoute