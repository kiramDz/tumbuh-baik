import { AxiosError } from "axios";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { NextRequest } from "next/server";

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
    bmkg: "bmkg",
    satelit: "satelite",
    buoys: "buoys",
  };

  return categoryMap[slug] || slug; // Default jika slug tidak dikenali
}

export function absoluteUrl(req: NextRequest) {
  const protocol = req.headers.get("x-forwarded-proto") || "http";
  const host = req.headers.get("host");
  return `${protocol}://${host}`;
}
