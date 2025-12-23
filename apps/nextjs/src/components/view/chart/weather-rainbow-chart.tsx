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
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="time"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={(value) => value.slice(0, 5)} // jam:menit
            />
            <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
            <Line
              dataKey="temperature"
              type="bump"
              stroke="url(#colorGray)"
              dot={false}
              strokeWidth={2}
              filter="url(#gray-line-glow)"
            />
            <defs>
              <linearGradient id="colorGray" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#1f2937" stopOpacity={0.9} />
                <stop offset="20%" stopColor="#374151" stopOpacity={0.9} />
                <stop offset="40%" stopColor="#4b5563" stopOpacity={0.9} />
                <stop offset="60%" stopColor="#6b7280" stopOpacity={0.9} />
                <stop offset="80%" stopColor="#9ca3af" stopOpacity={0.9} />
                <stop offset="100%" stopColor="#d1d5db" stopOpacity={0.9} />
              </linearGradient>
              <filter id="gray-line-glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="10" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
