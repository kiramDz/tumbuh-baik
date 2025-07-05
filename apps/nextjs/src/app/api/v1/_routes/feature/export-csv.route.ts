// 1. API Route untuk Export CSV - /app/api/v1/export-csv/route.ts
import { Hono } from "hono";
import db from "@/lib/database/db";
import { BmkgData } from "@/lib/database/schema/dataset/bmkg.model";
import { BuoysData } from "@/lib/database/schema/dataset/buoys.model"; // Sesuaikan path
import { DailySummary } from "@/lib/database/schema/model/plantSummary.model";
import { parseError } from "@/lib/utils";
import { convertToCSV } from "@/lib/export";
import { extractColumnsFromDef } from "@/lib/column-extractor";
import { bmkgColumns } from "@/app/dashboard/_components/table/columns/bmkg-columns";
import { buoysColumns } from "@/app/dashboard/_components/table/columns/buoys-columns";
import { holtWinterColumns } from "@/app/dashboard/_components/kaltam/kaltam-table/column";
const exportRoute = new Hono();

// 🧠 Map kategori ke model, kolom, dan nama file
const categoryMap = {
  bmkg: {
    model: BmkgData,
    columns: bmkgColumns,
    filename: "bmkg_data",
  },
  buoys: {
    model: BuoysData,
    columns: buoysColumns,
    filename: "buoys_data",
  },
  kaltam: {
    model: DailySummary,
    columns: holtWinterColumns,
    filename: "kaltam_data",
  },
};

// Function untuk convert data ke CSV format
exportRoute.get("/", async (c) => {
  try {
    console.log("=== Export CSV Endpoint terpanggil ===");
    await db();

    const category = c.req.query("category"); // bmkg atau buoys
    const sortBy = c.req.query("sortBy") || "Date";
    const sortOrder = c.req.query("sortOrder") || "desc";

    const config = categoryMap[category as keyof typeof categoryMap];
    if (!config) {
      return c.json(
        {
          message: "Error",
          description: "Invalid category. Must be one of: " + Object.keys(categoryMap).join(", "),
          data: null,
        },
        { status: 400 }
      );
    }

    const sortQuery: Record<string, 1 | -1> = {
      [sortBy]: sortOrder === "desc" ? -1 : 1,
    };

    const data = await config.model.find().sort(sortQuery).lean();
    const columnInfo = extractColumnsFromDef(config.columns);
    const csvContent = convertToCSV(data, columnInfo);

    c.header("Content-Type", "text/csv;charset=utf-8;");
    c.header("Content-Disposition", `attachment; filename="${config.filename}_${new Date().toISOString().split("T")[0]}.csv"`);

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