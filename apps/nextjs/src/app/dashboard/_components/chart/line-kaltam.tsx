"use client";

import { CartesianGrid, Line, LineChart, XAxis, ResponsiveContainer } from "recharts";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { TrendingDown } from "lucide-react";

const chartData = [
  { month: "January", desktop: 186 },
  { month: "February", desktop: 305 },
  { month: "March", desktop: 237 },
  { month: "April", desktop: 73 },
  { month: "May", desktop: 209 },
  { month: "June", desktop: 214 },
];

const chartConfig = {
  desktop: {
    label: "Desktop",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig;

export function RainbowGlowGradientLineChart() {
  return (
    <div className="w-2/3 l aspect-square max-h-[250px]">
      <div className="flex flex-col">
        <h3 className="text-lg font-semibold flex items-center">
          Rainbow Line Chart
          <span className="ml-2 inline-flex items-center gap-1 rounded-md border-none bg-red-500/10 px-2 py-0.5 text-sm text-red-500">
            <TrendingDown className="h-4 w-4" />
            <span>-5.2%</span>
          </span>
        </h3>
        <p className="text-sm text-muted-foreground">January - June 2024</p>
      </div>

      <div className="mt-4">
        <ChartContainer config={chartConfig} className="w-full">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData} margin={{ left: 4, right: 4 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="month" tickLine={false} axisLine={false} tickMargin={8} tickFormatter={(value) => value.slice(0, 3)} />
              <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
              <Line dataKey="desktop" type="bump" stroke="url(#colorUv)" dot={false} strokeWidth={2} filter="url(#rainbow-line-glow)" />
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
          </ResponsiveContainer>
        </ChartContainer>
      </div>
    </div>
  );
}
