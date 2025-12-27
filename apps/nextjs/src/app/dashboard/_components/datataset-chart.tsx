"use client";

import { GetChartDataBySlug } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertCircle, TrendingUp, Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface ChartSectionProps {
  collectionName: string;
}

export default function ChartSection({ collectionName }: ChartSectionProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["chart-data", collectionName],
    queryFn: () => GetChartDataBySlug(collectionName),
  });

  const [selectedColumn, setSelectedColumn] = useState<string>("");

  if (data && data.numericColumns?.length > 0 && !selectedColumn) {
    setSelectedColumn(data.numericColumns[0]);
  }

  if (isLoading) {
    return (
      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Loader2 className="h-5 w-5 animate-spin" />
            Memuat Chart...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center">
            <div className="text-muted-foreground">Sedang memuat data chart...</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className="mt-6">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>Gagal memuat data chart. Silakan coba lagi.</AlertDescription>
      </Alert>
    );
  }

  if (!data || !data.items?.length || !data.dateColumn) {
    return (
      <Alert className="mt-6 border-yellow-500">
        <AlertCircle className="h-4 w-4 text-yellow-600" />
        <AlertTitle className="text-yellow-600">Data Tidak Dapat Ditampilkan</AlertTitle>
        <AlertDescription>
          Dataset ini tidak memiliki kolom <strong>Date</strong> atau tidak ada data yang bisa ditampilkan.
        </AlertDescription>
      </Alert>
    );
  }

  if (!data.numericColumns?.length) {
    return (
      <Alert className="mt-6 border-yellow-500">
        <AlertCircle className="h-4 w-4 text-yellow-600" />
        <AlertTitle className="text-yellow-600">Data Tidak Dapat Ditampilkan</AlertTitle>
        <AlertDescription>Dataset ini tidak memiliki kolom numerik yang bisa ditampilkan.</AlertDescription>
      </Alert>
    );
  }

  const chartData = data.items.map((item: any) => ({
    date: item[data.dateColumn],
    value: Number(item[selectedColumn]) || 0,
  }));

  const chartConfig = {
    value: {
      label: selectedColumn,
      color: "hsl(var(--chart-1))",
    },
  } satisfies ChartConfig;

  return (
    <Card className="mt-6">
      <CardHeader>
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Visualisasi Data
            </CardTitle>
            <CardDescription className="mt-1">Menampilkan {chartData.length} data point</CardDescription>
          </div>
          <Select value={selectedColumn} onValueChange={setSelectedColumn}>
            <SelectTrigger className="w-full sm:w-[250px]">
              <SelectValue placeholder="Pilih kolom untuk chart" />
            </SelectTrigger>
            <SelectContent>
              {data.numericColumns.map((column: string) => (
                <SelectItem key={column} value={column}>
                  {column}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>

      <CardContent>
        <ChartContainer config={chartConfig}>
          <LineChart accessibilityLayer data={chartData} margin={{ left: 12, right: 12 }}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              minTickGap={32}
              tickFormatter={(value) => {
                if (typeof value === "string" && value.includes("-")) {
                  const date = new Date(value);
                  return date.toLocaleDateString("id-ID", {
                    month: "short",
                    day: "numeric",
                  });
                }
                return value;
              }}
            />
            <YAxis domain={["dataMin - 3", "dataMax + 3"]} />

            <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />

            <Line dataKey="value" type="linear" stroke="#2563eb" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
