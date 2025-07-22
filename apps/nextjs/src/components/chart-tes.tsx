"use client";

import * as React from "react";
import { Area, AreaChart, XAxis } from "recharts";

import { Card, CardContent } from "@/components/ui/card";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";


interface WeatherChartProps {
  hourlyForecast: { time: string; temperature: number; weather: string }[];
}

const chartConfig = {
  visitors: {
    label: "Visitors",
  },
  desktop: {
    label: "Desktop",
    color: "var(--chart-1)",
  },
  mobile: {
    label: "Mobile",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig;

export function WeatherChart({ hourlyForecast }: WeatherChartProps) {
  const chartConfig = {
    temperature: {
      label: "Temperature (°C)",
      color: "hsl(var(--chart-1))",
    },
  };
  return (
    <Card className="pt-0 border-none shadow-none">
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer config={chartConfig} className="aspect-auto h-[250px] w-full">
          <AreaChart data={hourlyForecast}>
            <defs>
              <linearGradient id="fillTemperature" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-temperature)" stopOpacity={0.8} />
                <stop offset="95%" stopColor="var(--color-temperature)" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="time"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              minTickGap={32}
              tickFormatter={(value) => value} // Time is already formatted as HH:00
            />
            <ChartTooltip cursor={false} content={<ChartTooltipContent labelFormatter={(value) => value} indicator="dot" formatter={(value, name) => [value, "Temperature"]} />} />
            <Area dataKey="temperature" type="natural" fill="url(#fillTemperature)" stroke="var(--color-temperature)" />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
