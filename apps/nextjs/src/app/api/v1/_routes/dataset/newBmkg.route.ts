import { Hono } from "hono";
import axios from "axios";
import { parseError } from "@/lib/utils";
import { BMKGApi } from "@/lib/database/schema/dataset/bmkgApi.model";

const gampongList = [
  "11.06.02.2001", // aceh besar
  "11.71.04.2005", // banda aceh
  "11.07.16.2001", // pidie
  "11.14.05.2012", // Aceh jaya
];

const bmkgLiveRoute = new Hono();

bmkgLiveRoute.get("/all", async (c) => {
  try {
    const results: any[] = [];

    for (const kodeGampong of gampongList) {
      const response = await axios.get(`https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=${kodeGampong}`, { timeout: 10000 });

      const data = response.data;
      const cuacaNested = data.data?.[0]?.cuaca ?? [];
      const cuacaData = cuacaNested.flat();

      const mappedCuaca = cuacaData.map((item: any) => ({
        local_datetime: item.local_datetime,
        t: item.t,
        hu: item.hu,
        weather_desc: item.weather_desc,
        ws: item.ws,
        wd: item.wd,
        tcc: item.tcc,
        vs_text: item.vs_text,
      }));

      // Bentuk sesuai schema BMKGApi (meski tidak disimpan ke DB)
      const bmkgDoc = new BMKGApi({
        kode_gampong: kodeGampong,
        nama_gampong: data.lokasi?.desa || "Unknown",
        tanggal_data: new Date().toISOString().split("T")[0],
        analysis_date: new Date(data.data?.[0]?.cuaca?.[0]?.analysis_date || new Date()),
        data: mappedCuaca,
      });

      // Konversi jadi JSON plain (biar sama kayak hasil dari Mongo)
      results.push(bmkgDoc.toObject());
    }

    return c.json({
      message: "Success",
      description: "Realtime data from BMKG",
      data: results,
    });
  } catch (error) {
    console.error("Error fetching data from BMKG:", error);
    const err = parseError(error);
    return c.json({ message: "Error", description: err }, { status: 500 });
  }
});

export default bmkgLiveRoute;