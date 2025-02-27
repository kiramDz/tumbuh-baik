import { AxiosError } from "axios";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import Papa from "papaparse";

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

export function formatFileSize(sizeInBytes: number): string {
  if (sizeInBytes === 0) return "0 Bytes";

  const units = ["Bytes", "KB", "MB", "GB", "TB"];
  const k = 1024; // 1 KB = 1024 Bytes
  const i = Math.floor(Math.log(sizeInBytes) / Math.log(k)); // Determine the unit

  const size = (sizeInBytes / Math.pow(k, i)).toFixed(2); // Convert size to the appropriate unit
  return `${size} ${units[i]}`; // Combine size with unit
}

type FileCategory = "document" | "video" | "image" | "audio" | "other";

export function getCategoryFromMimeType(mimeType: string): FileCategory {
  if (mimeType === "application/pdf") return "document";

  if (mimeType.startsWith("video/")) return "video";

  if (mimeType.startsWith("image/")) return "image";

  if (mimeType.startsWith("audio/")) return "audio";

  return "other";
}

export function generatePageKey(page: string): string {
  if (page === "documents") return "document";

  if (page === "images") return "image";

  if (page === "videos") return "video";

  if (page === "others") return "other";

  return page;
}

// stroage slug
export function pageIdentifier(slug: string): string {
  const categoryMap: Record<string, string> = {
    "bmkg-station": "BMKG",
    "citra-satelit": "Citra Satelit",
    "temperatur-laut": "Temperatur Laut",
    "daily-weather": "Daily Weather",
  };

  return categoryMap[slug] || slug; // Default jika slug tidak dikenali
}

export function dynamicDownload(url: string, name: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = name;

  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

export function ActionResponse<T>(data: T): T {
  return JSON.parse(JSON.stringify(data));
}

export function parseCSV(csvString: string, columns: string[]) {
  try {
    const parsed = Papa.parse(csvString, { header: true });

    if (parsed.errors.length > 0) {
      throw new Error(`CSV Parsing Error: ${parsed.errors[0].message}`);
    }

    return parsed.data.map((row: any) => {
      const filteredRow: any = {};
      columns.forEach((col) => {
        filteredRow[col] = row[col] || null; // Jika kolom tidak ada, beri null
      });
      return filteredRow;
    });
  } catch (error) {
    return { error: `Failed to parse CSV: ${error instanceof Error ? error.message : String(error)}` };
  }
}
