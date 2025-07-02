import { AxiosError } from "axios";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import Papa from "papaparse";
import {  NextRequest } from "next/server";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function parseError(error: unknown) {
  if (error instanceof Error) return error.message;

  if (error instanceof Response) return error.statusText;

  if (error instanceof ErrorEvent) return error.message;

  if (error instanceof AxiosError) return error.response?.data?.message || error.message;

  if (typeof error === "string") return error;

  return JSON.stringify(error);
}

// stroage slug
export function pageIdentifier(slug: string): string {
  const categoryMap: Record<string, string> = {
    "bmkg": "bmkg",
    "satelit": "satelite",
    "buoys": "buoys",
  };

  return categoryMap[slug] || slug; // Default jika slug tidak dikenali
}

export function absoluteUrl(req: NextRequest) {
  const protocol = req.headers.get("x-forwarded-proto") || "http";
  const host = req.headers.get("host");
  return `${protocol}://${host}`;
}



export function parseCSV(csvString: string, columns: string[]) {
  try {
    // Tambahkan opsi delimiter dan pastikan newline terdeteksi dengan benar
    const parsed = Papa.parse(csvString, {
      header: true,
      delimiter: ",", // Secara eksplisit tentukan delimiter
      newline: "\n", // Secara eksplisit tentukan newline
      skipEmptyLines: true, // Lewati baris kosong
    });

    console.log("Hasil Papa.parse:", parsed.data[0]); // Debug untuk melihat baris pertama

    if (parsed.errors.length > 0) {
      throw new Error(`CSV Parsing Error: ${parsed.errors[0].message}`);
    }

    return parsed.data.map((row: any) => {
      const filteredRow: any = {};
      columns.forEach((col) => {
        filteredRow[col] = row[col] || null;
      });
      return filteredRow;
    });
  } catch (error) {
    return { error: `Failed to parse CSV: ${error instanceof Error ? error.message : String(error)}` };
  }
}
