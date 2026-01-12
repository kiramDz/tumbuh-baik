import { parse } from "csv-parse/sync";

export async function parseFile({ fileBuffer, fileType }: { fileBuffer: Buffer; fileType: "csv" | "json" }): Promise<Record<string, any>[]> {
  let parsedData: Record<string, any>[];

  try {
    if (fileType === "csv") {
      const raw = parse(fileBuffer.toString(), {
        columns: true,
        skip_empty_lines: true,
        trim: true,
      });

      parsedData = raw.map((row: any) => {
        const parsedRow: Record<string, any> = {};
        for (const key in row) {
          parsedRow[key] = parseValue(row[key]);
        }
        return parsedRow;
      });
    } else if (fileType === "json") {
      const raw = JSON.parse(fileBuffer.toString());
      if (!Array.isArray(raw)) throw new Error("Format JSON tidak valid");
      parsedData = raw.map((row) => {
        const parsedRow: Record<string, any> = {};
        for (const key in row) {
          parsedRow[key] = parseValue(row[key]);
        }
        return parsedRow;
      });
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
function parseValue(value: string): any {
  if (value === "" || value === null || value === undefined) return null;

  // Deteksi format tanggal (DD/MM/YYYY)
  const datePattern = /^\d{1,2}\/\d{1,2}\/\d{4}$/;
  if (datePattern.test(value)) {
    const [day, month, year] = value.split("/");
    const date = new Date(`${year}-${month}-${day}`);
    return isNaN(date.getTime()) ? value : date;
  }

  // Deteksi number
  const num = Number(value);
  if (!isNaN(num)) return num;

  return value;
}
