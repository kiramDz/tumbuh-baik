"use client";

import { CartesianGrid, Line, LineChart, XAxis, YAxis, Legend } from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartConfig } from "@/components/ui/chart";

interface WeatherLineChartProps {
  data: {
    time: string;
    value: number;
  }[];
  type: "temperature" | "cloud_cover" | "wind";
}

const chartConfig = {
  temperature: {
    label: "Temperature (°C)",
    color: "#FACC15", // kuning
  },
  cloud_cover: {
    label: "Tutupan Awan (%)",
    color: "#3B82F6", // biru
  },
  wind_speed: {
    label: "Wind Speed (km/h)",
    color: "#22C55E", // hijau
  },
} satisfies ChartConfig;

export function WeatherLineChart({ data, type }: WeatherLineChartProps) {
  const chartMeta = {
    temperature: {
      color: "#FACC15", // kuning
      label: "Temperature (°C)",
    },
    cloud_cover: {
      color: "#3B82F6", // biru
      label: "Tutupan Awan (%)",
    },
    wind: {
      color: "#22C55E", // hijau
      label: "Wind Speed (km/h)",
    },
  }[type];

  // Format waktu → "10:00", "13:00", dst
  const formatHour = (time: string) => {
    if (!time) return "-";
    const match = time.match(/(\d{1,2}):/);
    const hour = match ? match[1].padStart(2, "0") : time;
    return `${hour}:00`;
  };

  return (
    <Card className="py-2  border-none shadow-none">
      <CardContent className="p-0 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer config={chartConfig} className="aspect-auto  h-[250px] sm:h-[280px] md:h-[320px] w-full">
          <LineChart data={data} margin={{ left: 12, right: 12 }}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="time" tickLine={false} axisLine={false} tickMargin={8} tickFormatter={formatHour} className="text-xs sm:text-sm" />
            <YAxis tickLine={false} axisLine={false} className="text-xs sm:text-sm" />
            <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
            <Legend verticalAlign="top" height={32} wrapperStyle={{ fontSize: "14px" }} />

            <Line dataKey="value" type="monotone" stroke={chartMeta.color} dot={false} strokeWidth={2.5} name={chartMeta.label} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
