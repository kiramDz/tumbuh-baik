import React from "react";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartConfig } from "@/components/ui/chart";

type RainfallSummary = {
  year: number;
  avgRainfall: number; // nilai dari backend
};

const chartConfig = {
  rainfall: {
    label: "Curah Hujan",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig;

export function RainfallAreaChart({ collectionName }: { collectionName: string }) {
  const { data, isLoading, error } = useQuery<RainfallSummary[]>({
    queryKey: ["rainfall-summary", collectionName],
    queryFn: async () => {
      const res = await fetch(`/api/v1/dataset-meta/rainfall-summary?collection=${collectionName}`);
      if (!res.ok) throw new Error("Gagal memuat data ringkasan curah hujan");
      const json = await res.json();
      return json.data; // diasumsikan format: [{ year: 2005, avgRainfall: 123.45 }, ...]
    },
    enabled: !!collectionName,
    refetchOnWindowFocus: false,
  });

  const chartData =
    data?.map((item) => ({
      year: item.year,
      rainfall: item.avgRainfall,
    })) ?? [];

  const totalRainfall = chartData.reduce((sum, item) => sum + item.rainfall, 0);

  if (isLoading) return <p>Loading chart...</p>;
  if (error) return <p>Error loading chart: {error.message}</p>;
  if (!chartData.length) return <p>No rainfall data found</p>;

  return (
    <Card className="pt-0">
      <CardHeader className="flex gap-2 space-y-0 border-b py-5">
        <div className="grid flex-1 gap-1">
          <CardTitle>Grafik Curah Hujan</CardTitle>
          <CardDescription>Total Rata-rata Curah Hujan: {totalRainfall.toFixed(2)} mm (2005–2025)</CardDescription>
        </div>
      </CardHeader>
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer config={chartConfig} className="aspect-auto h-[300px] w-full">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="fillRainfall" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-rainfall)" stopOpacity={0.8} />
                <stop offset="95%" stopColor="var(--color-rainfall)" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="year" tickLine={false} axisLine={false} tickMargin={8} minTickGap={32} tickFormatter={(value) => value.toString()} />
            <YAxis
              domain={[0, "auto"]}
              tickFormatter={(value) => value.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ".")}
              label={{
                value: "Curah Hujan (mm)",
                angle: -90,
                position: "insideLeft",
              }}
            />
            <ChartTooltip cursor={false} content={<ChartTooltipContent labelFormatter={(value) => value.toString()} valueFormatter={(value) => `${value.toLocaleString("id-ID")} mm`} indicator="dot" />} />
            <Area dataKey="rainfall" type="natural" fill="url(#fillRainfall)" stroke="var(--color-rainfall)" />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
