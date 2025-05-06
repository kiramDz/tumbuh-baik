import { Hono } from "hono";
import axios from "axios";
import { BMKGApi } from "@/lib/database/schema/bmkgApi.model";
import db from "@/lib/database/db";
import { parseError } from "@/lib/utils";

// Daftar kode wilayah gampong Aceh Besar (misalnya)
const gampongList = [
  "11.06.02.2001", // Mon Ikeun
  "11.06.02.2002", // Nusa
  // Tambahkan gampong lainnya
];

const bmkgFetcherRoute = new Hono();

bmkgFetcherRoute.get("/", async (c) => {
  try {
    await db();
    console.log("=== Fetching data from BMKG API ===");

    // Loop untuk fetch data dari BMKG per kode wilayah
    for (const kodeGampong of gampongList) {
      const response = await axios.get(`https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=${kodeGampong}`);

      const data = response.data; // Data dari BMKG
      console.log("Data from BMKG:", data);
      console.log("Isi cuaca:", data.data[0].cuaca);

      const cuacaNested = data.data?.[0]?.cuaca ?? [];
      const cuacaData = cuacaNested.flat();

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

      await BMKGApi.create({
        kode_gampong: kodeGampong, // dari loop
        nama_gampong: data.lokasi?.desa, // dari response
        tanggal_data: new Date().toISOString().split("T")[0],
        analysis_date: new Date(data.data?.[0]?.cuaca?.[0]?.analysis_date ?? new Date()),
        data: mappedCuaca,
      });

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

export default bmkgFetcherRoute;
