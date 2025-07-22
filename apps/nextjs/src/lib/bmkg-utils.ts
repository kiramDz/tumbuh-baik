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

export const getFinalConclusion = (
  dailyConclusion: WeatherConclusionResult,
  seasonalStatus: string
): WeatherConclusionResult & {
  seasonalStatus: string;
  finalStatus: string;
  note: string;
} => {
  let finalStatus = "";
  let note = "";

  if (seasonalStatus === "sangat cocok tanam" || seasonalStatus === "cocok tanam") {
    finalStatus = "Cocok Tanam";
    note = "Musim tanam dan cuaca mendukung.";
  } else if (seasonalStatus === "tidak cocok tanam") {
    finalStatus = "Belum Disarankan Tanam";
    note = "Cuaca bagus, tapi ini bukan musim tanam.";
  } else {
    finalStatus = "Perlu Dipertimbangkan";
    note = "Belum ada data musiman yang jelas.";
  }

  return {
    ...dailyConclusion,
    status: finalStatus.toLowerCase().includes("cocok") ? "cocok" : "tidak dianjurkan",
    seasonalStatus,
    finalStatus,
    note,
  };
};

export function getWeatherConclusionFromDailyData(todayData: BMKGDataItem[]): WeatherConclusionResult {
  if (!todayData || todayData.length === 0) {
    return {
      status: "waspada",
      reason: "Data cuaca hari ini tidak tersedia.",
      badge: ["Data Tidak Ada"],
      avg: { temperature: 0, humidity: 0, wind: 0, cloud: 0 },
    };
  }

  const tempSum = todayData.reduce((sum, d) => sum + d.t, 0);
  const humSum = todayData.reduce((sum, d) => sum + d.hu, 0);
  const windSum = todayData.reduce((sum, d) => sum + d.ws, 0);
  const cloudSum = todayData.reduce((sum, d) => sum + d.tcc, 0);

  const avgTemp = tempSum / todayData.length;
  const avgHum = humSum / todayData.length;
  const avgWind = windSum / todayData.length;
  const avgCloud = cloudSum / todayData.length;

  const weatherDescriptions = todayData.map((d) => d.weather_desc.toLowerCase());
  const hasExtremeWeather = weatherDescriptions.some((desc) => desc.includes("hujan lebat") || desc.includes("petir") || desc.includes("badai"));
  const hasStrongWind = todayData.some((d) => d.ws > 25);
  const hasExtremeTemp = todayData.some((d) => d.t > 35 || d.t < 20);

  const badges: string[] = [];
  if (avgTemp >= 24 && avgTemp <= 34) badges.push("Suhu Ideal");
  else badges.push("Suhu Tidak Stabil");

  if (avgHum >= 60 && avgHum <= 90) badges.push("Kelembapan Ideal");
  else badges.push("Kelembapan Tidak Stabil");

  if (avgWind <= 25) badges.push("Angin Normal");
  else badges.push("Angin Kencang");

  const dominantWeather = weatherDescriptions.filter((d) => d.includes("cerah")).length;
  if (dominantWeather >= todayData.length / 2) badges.push("Cerah");

  let status: WeatherConclusionResult["status"];
  let reason: string;

  if (hasExtremeWeather) {
    status = "tidak dianjurkan";
    reason = "Terdeteksi cuaca ekstrem seperti hujan lebat atau badai.";
  } else if (hasExtremeTemp || hasStrongWind) {
    status = "waspada";
    reason = hasExtremeTemp ? "Suhu hari ini terlalu ekstrem." : "Angin kencang terdeteksi hari ini.";
  } else {
    status = "cocok";
    reason = "Cuaca stabil dan mendukung aktivitas pertanian.";
  }

  return {
    status,
    reason,
    badge: badges,
    avg: {
      temperature: Number(avgTemp.toFixed(1)),
      humidity: Number(avgHum.toFixed(1)),
      wind: Number(avgWind.toFixed(1)),
      cloud: Number(avgCloud.toFixed(1)),
    },
  };
}

// untuk kartu kesimpula
export function getTodayWeatherConlusion(data: BMKGDataItem[]): BMKGDataItem[] {
  const today = new Date();
  const todayStr = today.toISOString().slice(0, 10); // format: 'YYYY-MM-DD'

  return data.filter((item) => {
    const datePart = item.local_datetime.slice(0, 10); // ambil 'YYYY-MM-DD'
    return datePart === todayStr;
  });
}

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

export const getChartData = (data: BMKGDataItem[]) => {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0);
  const end = new Date(today);
  end.setDate(end.getDate() + 2);
  end.setHours(23, 59, 59);

  return data
    .filter((item) => {
      const dt = new Date(item.local_datetime.replace(" ", "T"));
      return dt >= today && dt <= end;
    })
    .map((item) => ({
      time: new Date(item.local_datetime.replace(" ", "T")).getTime().toString(),
      temperature: item.t,
      humidity: item.hu,
    }));
};

// utils/weather-analysis.ts

export function getWindCondition(ws: number, wd: string): string {
  const windSpeed = ws; // dalam km/h
  const direction = wd.toUpperCase();

  if (direction === "Calm" || windSpeed < 5) {
    return "Angin tenang, sangat baik untuk aktivitas pertanian.";
  }

  if (windSpeed >= 5 && windSpeed <= 20) {
    return `Angin dari arah ${direction}, kecepatan normal (${windSpeed} km/h), baik untuk pertanian.`;
  }

  if (windSpeed > 20 && windSpeed <= 35) {
    return `Angin dari arah ${direction}, cukup kencang (${windSpeed} km/h), waspadai tanaman muda atau tinggi.`;
  }

  if (windSpeed > 35) {
    return `Angin dari arah ${direction}, sangat kencang (${windSpeed} km/h), potensi merusak tanaman – perlu perlindungan.`;
  }

  return "Data angin tidak tersedia atau tidak valid.";
}

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
  const days = [0, 1, 2]; // hari ini sampai 2 hari ke depan

  return days.map((offset) => {
    const date = addDays(now, offset);
    const dayName = format(date, "EEEE", { locale: id });

    const filtered = (data ?? []).filter((item) => {
      if (!item?.local_datetime) return false; // tambahkan ini
      const itemDate = new Date(item.local_datetime.replace(" ", "T"));
      return itemDate >= now && itemDate <= end;
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
