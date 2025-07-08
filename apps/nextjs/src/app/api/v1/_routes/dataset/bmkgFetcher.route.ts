import { Hono } from "hono";
import axios from "axios";
import { BMKGApi } from "@/lib/database/schema/dataset/bmkgApi.model";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";

const gampongList = [
  "11.06.02.2001", // Mon Ikeun
  "11.06.02.2002", // Nusa
];

const bmkgFetcherRoute = new Hono();

bmkgFetcherRoute.get("/", async (c) => {
  try {
    await db();
    console.log("=== Fetching data from BMKG API ===");

    // Loop untuk fetch data dari BMKG per kode wilayah
    for (const kodeGampong of gampongList) {
      const response = await axios.get(`https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=${kodeGampong}`, {
        timeout: 10000, // 10 detik
      });

      const data = response.data; // Data dari BMKG
      console.log("Data from BMKG:", data);
      console.log("Isi cuaca:", data.data[0].cuaca);

      const cuacaNested = data.data?.[0]?.cuaca ?? [];
      const cuacaData = cuacaNested.flat();
      const tanggalHariIni = new Date().toISOString().split("T")[0];
      //FIXME : Replace 'any' type with a proper TypeScript interface for cuacaData mapping
      const mappedCuaca = cuacaData
        .filter((item: any) => item.local_datetime && item.t && item.hu && item.weather_desc && item.ws && item.wd && item.tcc && item.vs_text)
        .map((item: any) => ({
          local_datetime: item.local_datetime,
          t: item.t,
          hu: item.hu,
          weather_desc: item.weather_desc,
          ws: item.ws,
          wd: item.wd,
          tcc: item.tcc,
          vs_text: item.vs_text,
        }));
      // data lama tdk akan terhaous ketika data baru diupdate
      await BMKGApi.findOneAndUpdate(
        {
          kode_gampong: kodeGampong,
          tanggal_data: tanggalHariIni,
        },
        {
          nama_gampong: data.lokasi?.desa || "Unknown",
          analysis_date: new Date(data.data?.[0]?.cuaca?.[0]?.analysis_date || new Date()),
          data: mappedCuaca,
        },
        {
          upsert: true,
          new: true,
        }
      );

      // Simpan ke MongoDB
    }

    return c.json({
      message: "Data fetched and saved to MongoDB successfully.",
      description: "",
    });
  } catch (error) {
    console.error("Error fetching data from BMKG:", error);
    const err = parseError(error);
    return c.json({ message: "Error", description: err }, { status: 500 });
  }
});

//update setiap 48 jam (2 hari sekli)
// cron.schedule("0 7 */2 * *", async () => {
//   console.log("=== [CRON] Scheduled fetch BMKG API ===");

//   try {
//     const fetchUrl = `http://localhost:3000/api/v1/bmkg-fetch`; // Ganti dengan full URL jika perlu
//     await axios.get(fetchUrl);
//     console.log("[CRON] BMKG fetch success.");
//   } catch (error) {
//     console.error("[CRON] Error fetching BMKG:", error);
//   }
// });

export default bmkgFetcherRoute;
