import { parse } from "csv-parse/sync";

export async function parseFile({ fileBuffer, fileType }: { fileBuffer: Buffer; fileType: "csv" | "json" }): Promise<Record<string, any>[]> {
  let parsedData: Record<string, any>[];

  try {
    if (fileType === "csv") {
      parsedData = parse(fileBuffer.toString(), {
        columns: true,
        skip_empty_lines: true,
        trim: true,
      });
    } else if (fileType === "json") {
      parsedData = JSON.parse(fileBuffer.toString());
      if (!Array.isArray(parsedData)) {
        throw new Error("Format JSON tidak valid, harus berupa array of objects.");
      }
    } else {
      throw new Error("Tipe file tidak didukung.");
    }
  } catch (err) {
    throw new Error("Gagal memparsing file: " + (err as Error).message);
  }

  if (!Array.isArray(parsedData) || parsedData.length === 0) {
    throw new Error("Data kosong atau format tidak valid.");
  }

  return parsedData;
}
