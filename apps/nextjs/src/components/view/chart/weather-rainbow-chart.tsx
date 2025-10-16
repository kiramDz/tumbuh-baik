"use client";

import { CartesianGrid, Line, LineChart, XAxis, YAxis, Legend } from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";

interface WeatherChartProps {
  hourlyForecast: {
    time: string; // misal "10", "13", dst
    temperature: number;
    rain?: number;
    wind_speed?: number;
  }[];
}

const chartConfig = {
  temperature: {
    label: "Temperature (°C)",
    color: "#FACC15", // kuning
  },
  rain: {
    label: "Rainfall (mm)",
    color: "#3B82F6", // biru
  },
  wind_speed: {
    label: "Wind Speed (km/h)",
    color: "#22C55E", // hijau
  },
} satisfies ChartConfig;

export function RainbowGlowGradientLineChart({ hourlyForecast }: WeatherChartProps) {
  // ✅ Format jam menjadi "10:00", "13:00", "01:00" dst
  const formatHour = (hour: string) => {
    if (!hour) return "-";
    // Jika input hanya "10" atau "1", buat jadi "10:00" / "01:00"
    if (/^\d{1,2}$/.test(hour)) {
      const h = hour.padStart(2, "0");
      return `${h}:00`;
    }
    // Jika format lain (misalnya "2025-10-14 10:00:00"), ambil jamnya
    const match = hour.match(/(\d{1,2}):(\d{2})/);
    return match ? `${match[1].padStart(2, "0")}:${match[2]}` : hour;
  };

  return (
    <Card className="pt-0 border-none shadow-none">
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer config={chartConfig} className="aspect-auto h-[300px] w-full">
          <LineChart
            data={hourlyForecast}
            margin={{
              left: 12,
              right: 12,
            }}
          >
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="time" tickLine={false} axisLine={false} tickMargin={8} tickFormatter={formatHour} />
            <YAxis yAxisId="left" orientation="left" tickLine={false} axisLine={false} tickMargin={4} domain={["auto", "auto"]} />
            <YAxis yAxisId="right" orientation="right" tickLine={false} axisLine={false} tickMargin={4} domain={["auto", "auto"]} />

            <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
            <Legend verticalAlign="top" height={36} />

            {/* Temperatur */}
            <Line yAxisId="left" dataKey="temperature" type="monotone" stroke={chartConfig.temperature.color} dot={false} strokeWidth={2.5} name={chartConfig.temperature.label} />
            {/* Curah Hujan */}
            <Line yAxisId="right" dataKey="rain" type="monotone" stroke={chartConfig.rain.color} dot={false} strokeWidth={2.5} name={chartConfig.rain.label} />
            {/* Kecepatan Angin */}
            <Line yAxisId="right" dataKey="wind_speed" type="monotone" stroke={chartConfig.wind_speed.color} dot={false} strokeWidth={2.5} name={chartConfig.wind_speed.label} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
