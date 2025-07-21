"use client";

import * as React from "react";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartConfig } from "@/components/ui/chart";

type RainfallRecord = {
  Date: string;
  Year: number;
  RR_imputed: number;
};

const chartConfig = {
  rainfall: {
    label: "Curah Hujan",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig;

export function RainfallAreaChart({ collectionName }: { collectionName: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["rainfall-chart", collectionName],
    queryFn: async () => {
      const res = await fetch(`/api/v1/dataset-meta/${collectionName}?sortBy=Year&sortOrder=asc&page=1&pageSize=10000`);
      const json = await res.json();
      return json.data.items as RainfallRecord[];
    },
    enabled: !!collectionName,
    refetchOnWindowFocus: false,
  });

  const chartData = React.useMemo(() => {
    if (!data) return [];

    // Buat daftar tahun dari 2005 hingga 2025
    const years = Array.from({ length: 2025 - 2005 + 1 }, (_, i) => 2005 + i);

    // Kelompokkan data berdasarkan tahun, hitung total RR_imputed per tahun
    const yearlyData = data
      .filter((item) => item.Year && typeof item.RR_imputed === "number")
      .reduce((acc, item) => {
        const year = item.Year;
        if (!acc[year]) {
          acc[year] = { total: 0, count: 0 };
        }
        acc[year].total += item.RR_imputed;
        acc[year].count += 1;
        return acc;
      }, {} as Record<number, { total: number; count: number }>);

    // Buat data untuk grafik, isi tahun yang kosong dengan 0
    return years.map((year) => ({
      year,
      rainfall: yearlyData[year] ? yearlyData[year].total : 0,
    }));
  }, [data]);

  if (isLoading) return <p>Loading chart...</p>;
  if (error) return <p>Error loading chart: {error.message}</p>;
  if (!chartData.length) return <p>No rainfall data found</p>;

  const totalRainfall = chartData.reduce((sum, item) => sum + item.rainfall, 0);

  return (
    <Card className="pt-0">
      <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5">
        <div className="grid flex-1 gap-1">
          <CardTitle>Grafik Curah Hujan</CardTitle>
          <CardDescription>Total Curah Hujan: {totalRainfall.toFixed(2)} mm (2005-2025)</CardDescription>
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
            <YAxis label={{ value: "Curah Hujan (mm)", angle: -90, position: "insideLeft" }} tickLine={false} axisLine={false} tickMargin={8} ticks={[0, 10, 50, 100]} domain={[0, "auto"]} />
            <ChartTooltip cursor={false} content={<ChartTooltipContent labelFormatter={(value) => value.toString()} valueFormatter={(value) => `${value.toFixed(2)} mm`} indicator="dot" />} />
            <Area dataKey="rainfall" type="natural" fill="url(#fillRainfall)" stroke="var(--color-rainfall)" />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
