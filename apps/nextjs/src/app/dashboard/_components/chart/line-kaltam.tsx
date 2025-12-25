"use client";

import { useQuery } from "@tanstack/react-query";
import { CartesianGrid, Line, LineChart, XAxis, YAxis, ResponsiveContainer } from "recharts";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { TrendingDown, TrendingUp } from "lucide-react";
import { getHoltWinterDaily } from "@/lib/fetch/files.fetch";

export const description = "Line charts with forecast values";

const chartConfig = {
  value: {
    label: "Forecast Value",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig;

export function RainbowGlowGradientLineChart() {
  const { data: rawData, isLoading } = useQuery({
    queryKey: ["hw-daily-full"],
    queryFn: () => getHoltWinterDaily(1, 365),
    refetchOnWindowFocus: false,
  });

  if (isLoading) return <p>Loading line charts...</p>;

  const items = rawData?.items || [];
  if (items.length === 0) return <p>No forecast data available.</p>;

  const parameters = new Set<string>();
  items.forEach((item: any) => {
    Object.keys(item.parameters || {}).forEach((param) => parameters.add(param));
  });

  const paramArray = Array.from(parameters);

  const groupedData: Record<string, Array<{ date: string; value: number }>> = {};
  paramArray.forEach((param) => {
    groupedData[param] = items
      .filter((item: any) => item.parameters?.[param]?.forecast_value != null)
      .map((item: any) => ({
        date: new Date(item.forecast_date).toLocaleDateString("en-US", { month: "short" }),
        value: item.parameters[param].forecast_value,
      }))
      .sort((a: any, b: any) => new Date(a.date).getTime() - new Date(b.date).getTime());
  });

  return (
    <div className="w-full flex flex-col gap-4">
      {" "}
      {paramArray.map((param, index) => {
        const chartData = groupedData[param];
        if (chartData.length === 0) return null;

        const firstValue = chartData[0].value;
        const lastValue = chartData[chartData.length - 1].value;
        const percentChange = ((lastValue - firstValue) / firstValue) * 100;
        const isDown = percentChange < 0;
        const trendColor = isDown ? "bg-red-500/10 text-red-500" : "bg-green-500/10 text-green-500";
        const TrendIcon = isDown ? TrendingDown : TrendingUp;

        return (
          <div key={index} className="flex flex-col rounded-2xl bg-background p-4 aspect-video max-h-[300px]">
            <div className="flex flex-col">
              <h3 className="text-lg font-semibold flex items-center">
                {param}
              
              </h3>
            </div>

            <div className="mt-4">
              <ChartContainer config={chartConfig} className="w-full h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="date" tickLine={false} axisLine={false} tickMargin={8} interval="preserveStartEnd" />
                    <YAxis hide={false} stroke="#2563eb" />
                    <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                    <Line dataKey="value" type="monotone" stroke="#2563eb" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartContainer>
            </div>
          </div>
        );
      })}
    </div>
  );
}
