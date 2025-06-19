import { Hono } from "hono";
import db from "@/lib/database/db"; // Sesuaikan path sesuai struktur proyek
import { File } from "@/lib/database/schema/file.model";
import { pinata } from "@/lib/pinata/config";
import { parseCSV } from "@/lib/utils";

const visualizationRoute = new Hono();

// Endpoint untuk fetch data daily-weather
// cara tes => localhost:3000/api/v1/visualization/daily-weather
visualizationRoute.get("/daily-weather", async (c) => {
  await db(); // Pastikan koneksi ke database

  console.log("=== Endpoint /daily-weatjer terpanggil ===");
  // Ambil semua file dengan kategori 'daily-weather'

  // console.log(await File.find({ category: "Daily Weather" }));
  const files = await File.find({ category: "Daily Weather" }).lean();

  if (!files.length) return c.json({ error: "No files found" }, 404);

  // Cari file bernama "dw-data.csv"
  const targetFile = files.find((file) => file.name === "dw-data.csv");

  if (!targetFile) return c.json({ error: "File 'dw-data.csv' not found" }, 404);
  try {
    // Fetch CSV dari Pinata
    // g kayak biasa, yg kita cuman perlu metadata makanya cuman fetch dari mongodb
    // Ambil data CSV dari Pinata
    const response = await pinata.gateways.get(targetFile.cid);
    console.log("Response dari Pinata:", response);
    const csvData = await response.data;

    // Parse CSV ke JSON
    const parsedData = parseCSV(csvData, ["timestamp", "temperature"]);
    console.log("Hasil parsed:", parsedData);

    return c.json({ file: targetFile, data: parsedData });
  } catch (error) {
    return c.json({ error: `Failed to fetch file: ${error.message}` }, 500);
  }
});

export default visualizationRoute;
