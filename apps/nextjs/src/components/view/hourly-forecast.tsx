"use client";

import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import WeatherIcon from "./weather-icon";

interface HourlyForecastItem {
  local_datetime?: string;
  time?: string;
  t?: number;
  temperature?: number;
  hu?: number;
  humidity?: number;
  rain?: number; // curah hujan opsional
  weather_desc?: string;
  weather?: string;
}

interface HourlyForecastListProps {
  hourlyForecast: HourlyForecastItem[];
}

export default function HourlyForecastList({ hourlyForecast }: HourlyForecastListProps) {
  const getCardColor = (desc?: string) => {
    if (!desc) return "bg-slate-200/20 border-slate-300/40";
    if (desc.toLowerCase().includes("hujan")) return "bg-blue-500/20 border-blue-500/40";
    if (desc.toLowerCase().includes("berawan")) return "bg-gray-400/20 border-gray-400/40";
    if (desc.toLowerCase().includes("cerah")) return "bg-yellow-400/20 border-yellow-400/40";
    return "bg-slate-200/20 border-slate-300/40";
  };

  // ✅ Format jam dari "10" → "10.00 WIB" atau dari ISO datetime → "13.00 WIB"
  const formatHour = (datetime?: string) => {
    if (!datetime) return "-";

    // jika cuma angka (misal "10" atau "13"), jangan pakai new Date()
    if (/^\d{1,2}$/.test(datetime)) {
      return `${datetime.padStart(2, "0")}.00 WIB`;
    }

    // jika format ISO (misalnya "2025-10-14T10:00:00+07:00")
    const d = new Date(datetime);
    if (isNaN(d.getTime())) return "-";

    return d.toLocaleTimeString("id-ID", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  };

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold mb-2">Prakiraan 3 Jam ke Depan</h3>

      <ScrollArea className="w-full whitespace-nowrap rounded-md border">
        <div className="flex w-max space-x-4 p-4">
          {hourlyForecast.map((item, idx) => {
            const desc = item.weather_desc || item.weather || "";
            const temp = item.t ?? item.temperature ?? "-";
            const hum = item.hu ?? item.humidity ?? "-";
            const rain = item.rain ?? "-";
            const time = item.local_datetime || item.time || "";

            return (
              <Card key={idx} className={`min-w-[120px] text-center border rounded-2xl shadow-sm transition-colors ${getCardColor(desc)}`}>
                <CardContent className="flex flex-col items-center justify-center p-3 space-y-2">
                  <span className="text-sm font-medium">{formatHour(time)}</span>
                  <WeatherIcon description={desc} />
                  <div className="text-sm capitalize">{desc}</div>
                  <div className="text-base font-semibold">{temp}°C</div>
                  <div className="text-xs text-muted-foreground">{hum}% RH</div>
                  {rain !== "-" && <div className="text-xs text-blue-500">{rain} mm</div>}
                </CardContent>
              </Card>
            );
          })}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}
