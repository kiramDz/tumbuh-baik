"use client";

import { CartesianGrid, Line, LineChart, XAxis } from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";

interface WeatherChartProps {
  hourlyForecast: { time: string; temperature: number; weather: string }[];
}

const chartConfig = {
  temperature: {
    label: "Temperature (°C)",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig;

export function RainbowGlowGradientLineChart({ hourlyForecast }: WeatherChartProps) {
  return (
    <Card className="pt-0 border-none shadow-none">
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer config={chartConfig} className="aspect-auto h-[250px] w-full">
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
            <Line dataKey="temperature" type="bump" stroke="url(#colorUv)" dot={false} strokeWidth={2} filter="url(#rainbow-line-glow)" />
            <defs>
              <linearGradient id="colorUv" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#0B84CE" stopOpacity={0.8} />
                <stop offset="20%" stopColor="#224CD1" stopOpacity={0.8} />
                <stop offset="40%" stopColor="#3A11C7" stopOpacity={0.8} />
                <stop offset="60%" stopColor="#7107C6" stopOpacity={0.8} />
                <stop offset="80%" stopColor="#C900BD" stopOpacity={0.8} />
                <stop offset="100%" stopColor="#D80155" stopOpacity={0.8} />
              </linearGradient>
              <filter id="rainbow-line-glow" x="-20%" y="-20%" width="140%" height="140%">
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
