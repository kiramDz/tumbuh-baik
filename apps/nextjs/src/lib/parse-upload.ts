import { parse } from "csv-parse/sync";
import mongoose from "mongoose";

export async function parseAndSaveFile({ fileBuffer, fileType, collectionTarget }: { fileBuffer: Buffer; fileType: "csv" | "json"; collectionTarget: string }) {
  let parsedData: any[];

  // Parse file sesuai jenisnya
  if (fileType === "csv") {
    parsedData = parse(fileBuffer.toString(), {
      columns: true,
      skip_empty_lines: true,
    });
  } else if (fileType === "json") {
    parsedData = JSON.parse(fileBuffer.toString());
    if (!Array.isArray(parsedData)) {
      throw new Error("Format JSON tidak valid, harus berupa array of objects.");
    }
  } else {
    throw new Error("Tipe file tidak didukung.");
  }

  // Simpan ke MongoDB ke collection dynamic
  const DynamicModel = mongoose.model(collectionTarget, new mongoose.Schema({}, { strict: false }), collectionTarget);
  await DynamicModel.insertMany(parsedData);

  return {
    message: `Berhasil menyimpan ${parsedData.length} data ke ${collectionTarget}`,
  };
}
