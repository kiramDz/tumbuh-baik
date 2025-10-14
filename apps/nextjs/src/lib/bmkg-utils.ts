import type { BMKGDataItem, BMKGApiData } from "@/types/table-schema";
import { format, addDays } from "date-fns";
import { id } from "date-fns/locale";

export interface WeatherConclusionResult {
  status: "cocok" | "waspada" | "tidak dianjurkan";
  reason: string;
  badge: string[];
  avg: {
    temperature: number;
    humidity: number;
    wind: number;
    cloud: number;
  };
}

// untuk kartu kesimpula

export const getTodayWeather = (data: BMKGDataItem[]) => {
  if (data.length === 0) return null;
  const now = new Date();
  return data.reduce((closest: BMKGDataItem, item: BMKGDataItem) => {
    const itemDate = new Date(item.local_datetime.replace(" ", "T"));
    const closestDate = new Date(closest.local_datetime.replace(" ", "T"));
    return Math.abs(itemDate.getTime() - now.getTime()) < Math.abs(closestDate.getTime() - now.getTime()) ? item : closest;
  });
};

export const getUniqueGampongData = (data: BMKGApiData[]) => {
  const map = new Map<string, BMKGApiData>();

  data.forEach((item) => {
    const key = item.kode_gampong;

    // Ambil data dengan tanggal hari ini jika tersedia
    const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD

    const isToday = item.tanggal_data === today;
    const isExistingToday = map.get(key)?.tanggal_data === today;

    // Kalau belum ada atau yang sekarang adalah data hari ini
    if (!map.has(key) || (isToday && !isExistingToday)) {
      map.set(key, item);
    }
  });

  return Array.from(map.values());
};

// utils/weather-analysis.ts

export const getHourlyForecastData = (data: BMKGDataItem[], unit: "metric" | "imperial" = "metric") => {
  const now = new Date();
  const end = new Date();
  end.setDate(end.getDate() + 1);
  end.setHours(23, 59, 59);

  return data
    .filter((item) => {
      const dt = new Date(item.local_datetime.replace(" ", "T"));
      return dt >= now && dt <= end;
    })
    .map((item) => ({
      time: new Date(item.local_datetime.replace(" ", "T")).toLocaleTimeString([], {
        hour: "2-digit",
        hour12: false,
      }),
      temperature: unit === "imperial" ? (item.t * 9) / 5 + 32 : item.t, // Convert to Fahrenheit if imperial
      weather: item.weather_desc?.toLowerCase() || "clear",
    }));
};

export type ForecastDay = {
  day: string;
  condition: string;
  temperature: number;
  icon: string;
};
export const getDailyForecastData = (data: BMKGDataItem[]): ForecastDay[] => {
  const now = new Date();
  const days = [0, 1, 2];

  return days.map((offset) => {
    const date = addDays(now, offset);
    const dayName = format(date, "EEEE", { locale: id });

    const start = date;
    const end = new Date(date);
    end.setHours(23, 59, 59, 999);

    const filtered = (data ?? []).filter((item) => {
      if (!item?.local_datetime) return false;
      const itemDate = new Date(item.local_datetime.replace(" ", "T"));
      return itemDate >= start && itemDate <= end;
    });

    const avgTemp = filtered.reduce((sum, item) => sum + (item.t || 0), 0) / (filtered.length || 1);

    const conditions = filtered.map((item) => item.weather_desc?.toLowerCase() || "");
    const mostFrequent = getMostFrequent(conditions);

    return {
      day: dayName,
      condition: capitalizeFirstLetter(mostFrequent),
      temperature: Math.round(avgTemp),
      icon: getWeatherEmoji(mostFrequent),
    };
  });
};

function getWeatherEmoji(desc: string) {
  if (desc.includes("hujan")) return "🌧️";
  if (desc.includes("berawan")) return "☁️";
  if (desc.includes("cerah")) return "☀️";
  return "☀️"; // default
}

// Fungsi bantu: cari yang paling sering muncul
function getMostFrequent(arr: string[]): string {
  const freq: Record<string, number> = {};
  arr.forEach((item) => {
    freq[item] = (freq[item] || 0) + 1;
  });

  return Object.entries(freq).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
}

// Fungsi bantu: kapitalisasi
function capitalizeFirstLetter(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
