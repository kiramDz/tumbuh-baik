import type { BMKGDataItem, BMKGApiData } from "@/types/table-schema";

export const getTodayWeather = (data: BMKGDataItem[]) => {
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

export const getHourlyForecastData = (data: BMKGDataItem[]) => {
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
      temperature: item.t,
      weather: item.weather_desc?.toLowerCase() || "clear",
    }));
};
