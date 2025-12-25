"use client";

import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
  Area,
  AreaChart,
  Scatter,
  ScatterChart,
  ReferenceLine,
} from "recharts";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Calendar, Activity, TrendingUp, Waves, Zap } from "lucide-react";
import {
  GetDecompositionDataBySlug,
  GetAvailableDecompositionParameters,
  DecompositionData,
} from "@/lib/fetch/files.fetch";
import { useMemo } from "react";

const chartConfig = {
  original: {
    label: "Original",
    color: "hsl(var(--chart-1))",
  },
  trend: {
    label: "Trend",
    color: "hsl(var(--chart-2))",
  },
  seasonal: {
    label: "Seasonal",
    color: "hsl(var(--chart-3))",
  },
  residual: {
    label: "Residual",
    color: "hsl(var(--chart-4))",
  },
} satisfies ChartConfig;

// Parameter labels and units
const paramLabels: Record<string, string> = {
  T2M: "Suhu Udara (2m)",
  T2M_MAX: "Suhu Maksimum",
  T2M_MIN: "Suhu Minimum",
  RH2M: "Kelembaban Udara",
  PRECTOTCORR: "Curah Hujan",
  ALLSKY_SFC_SW_DWN: "Radiasi Matahari",
  WS10M: "Kecepatan Angin",
  WS10M_MAX: "Kecepatan Angin Maksimum",
  WD10M: "Arah Angin",
};

const paramUnits: Record<string, string> = {
  T2M: "°C",
  T2M_MAX: "°C",
  T2M_MIN: "°C",
  RH2M: "%",
  PRECTOTCORR: "mm",
  ALLSKY_SFC_SW_DWN: "MJ/m²/day",
  WS10M: "m/s",
  WS10M_MAX: "m/s",
  WD10M: "degrees",
};

function getParamLabel(param: string): string {
  return paramLabels[param] || param;
}

function getParamUnit(param: string): string {
  return paramUnits[param] || "";
}

function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-10 w-64" />
      <div className="flex gap-2">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="h-9 w-32 rounded-lg" />
        ))}
      </div>
      {[...Array(4)].map((_, i) => (
        <Card key={i}>
          <CardHeader>
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-32" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-[220px] w-full rounded-lg" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function EmptyState({ message }: { message?: string }) {
  return (
    <Card className="border-dashed">
      <CardContent className="flex flex-col items-center justify-center py-16">
        <Activity className="h-12 w-12 text-muted-foreground mb-4" />
        <p className="font-medium text-lg">
          {message || "Tidak ada data dekomposisi"}
        </p>
        <p className="text-sm text-muted-foreground text-center max-w-sm mt-1">
          Pastikan dataset telah dipreprocessing terlebih dahulu
        </p>
      </CardContent>
    </Card>
  );
}

interface DecompositionChartProps {
  collectionName: string;
}

interface ChartDataPoint {
  date: string;
  fullDate: string;
  original: number;
  trend: number;
  seasonal: number;
  residual: number;
  year: number;
}

function ParameterDecompositionView({
  param,
  collectionName,
}: {
  param: string;
  collectionName: string;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["decomposition", collectionName, param],
    queryFn: () => GetDecompositionDataBySlug(collectionName, param),
    refetchOnWindowFocus: false,
    retry: 1,
  });

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(4)].map((_, i) => (
          <Card key={i}>
            <CardHeader>
              <Skeleton className="h-6 w-48" />
              <Skeleton className="h-4 w-32" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-[220px] w-full rounded-lg" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>
          {error instanceof Error
            ? error.message
            : "Gagal memuat data dekomposisi"}
        </AlertDescription>
      </Alert>
    );
  }

  if (!data) {
    return (
      <EmptyState
        message={`Tidak ada data dekomposisi untuk ${getParamLabel(param)}`}
      />
    );
  }

  // Transform data for charts
  const chartData: ChartDataPoint[] = data.decomposition.dates.map(
    (date, index) => {
      const dateObj = new Date(date);
      return {
        date: dateObj.getFullYear().toString(),
        fullDate: dateObj.toLocaleDateString("id-ID", {
          weekday: "long",
          year: "numeric",
          month: "long",
          day: "numeric",
        }),
        original: data.decomposition.original[index],
        trend: data.decomposition.trend[index],
        seasonal: data.decomposition.seasonal[index],
        residual: data.decomposition.residual[index],
        year: dateObj.getFullYear(),
      };
    }
  );

  // Calculate statistics
  const stats = {
    trend: {
      min: Math.min(...data.decomposition.trend.filter((v) => v !== null)),
      max: Math.max(...data.decomposition.trend.filter((v) => v !== null)),
    },
    seasonal: {
      min: Math.min(...data.decomposition.seasonal.filter((v) => v !== null)),
      max: Math.max(...data.decomposition.seasonal.filter((v) => v !== null)),
    },
    residual: {
      min: Math.min(...data.decomposition.residual.filter((v) => v !== null)),
      max: Math.max(...data.decomposition.residual.filter((v) => v !== null)),
    },
  };

  const unit = getParamUnit(param);
  const years = [...new Set(chartData.map((d) => d.year))].sort();
  const yearRange = years.length;
  const tickInterval =
    yearRange > 20
      ? Math.ceil(chartData.length / 10)
      : yearRange > 10
      ? Math.ceil(chartData.length / 15)
      : Math.ceil(chartData.length / 20);

  return (
    <div className="space-y-4">
      {/* Header Card with Metadata */}
      <Card>
        <CardHeader className="pb-4">
          <div className="space-y-1">
            <CardTitle className="text-xl font-semibold">
              {getParamLabel(param)} - Analisis Dekomposisi
            </CardTitle>
            <CardDescription className="flex items-center gap-2">
              <Calendar className="h-3.5 w-3.5" />
              Data historis ({data.metadata.dateRange.start} -{" "}
              {data.metadata.dateRange.end})
            </CardDescription>
          </div>

          {/* Statistics Summary */}
          <div className="grid grid-cols-3 gap-4 pt-4">
            <div className="space-y-1">
              <p className="text-xs font-medium flex items-center gap-1.5 text-[hsl(var(--chart-2))]">
                <TrendingUp className="h-3.5 w-3.5" />
                Trend
              </p>
              <p className="text-sm font-medium tabular-nums">
                [{stats.trend.min.toFixed(2)}, {stats.trend.max.toFixed(2)}]{" "}
                {unit}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-xs font-medium flex items-center gap-1.5 text-[hsl(var(--chart-3))]">
                <Waves className="h-3.5 w-3.5" />
                Seasonal
              </p>
              <p className="text-sm font-medium tabular-nums">
                [{stats.seasonal.min.toFixed(2)},{" "}
                {stats.seasonal.max.toFixed(2)}] {unit}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-xs font-medium flex items-center gap-1.5 text-[hsl(var(--chart-4))]">
                <Zap className="h-3.5 w-3.5" />
                Residual
              </p>
              <p className="text-sm font-medium tabular-nums">
                [{stats.residual.min.toFixed(2)},{" "}
                {stats.residual.max.toFixed(2)}] {unit}
              </p>
            </div>
          </div>

          {/* Model Info */}
          <div className="flex gap-2 pt-2">
            <Badge variant="outline" className="text-xs">
              Model: {data.metadata.model}
            </Badge>
            <Badge variant="outline" className="text-xs">
              Period: {data.metadata.period} days
            </Badge>
            <Badge variant="outline" className="text-xs">
              Data Points: {data.metadata.dataPoints.toLocaleString()}
            </Badge>
          </div>
        </CardHeader>
      </Card>

      {/* Chart 1: Original Series */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[hsl(var(--chart-1))]"></div>
            Original Series
          </CardTitle>
          <CardDescription className="text-xs">
            Raw time series data before decomposition
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[220px] w-full">
            <LineChart
              data={chartData}
              margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
                className="stroke-muted"
              />
              <XAxis
                dataKey="date"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                interval={tickInterval}
                className="text-xs"
                tick={{ fill: "hsl(var(--muted-foreground))" }}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                width={50}
                className="text-xs"
                tick={{ fill: "hsl(var(--muted-foreground))" }}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <ChartTooltip
                cursor={{
                  stroke: "hsl(var(--muted-foreground))",
                  strokeWidth: 1,
                }}
                content={
                  <ChartTooltipContent
                    formatter={(value) => [
                      `${Number(value).toFixed(3)} ${unit}`,
                      "Original",
                    ]}
                    labelFormatter={(label, payload) => {
                      if (payload?.[0]?.payload?.fullDate) {
                        return payload[0].payload.fullDate;
                      }
                      return label;
                    }}
                  />
                }
              />
              <Line
                type="monotone"
                dataKey="original"
                stroke="hsl(var(--chart-1))"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* Chart 2: Trend Component */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[hsl(var(--chart-2))]"></div>
            Trend Component
          </CardTitle>
          <CardDescription className="text-xs">
            Long-term progression pattern
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[220px] w-full">
            <LineChart
              data={chartData}
              margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
                className="stroke-muted"
              />
              <XAxis
                dataKey="date"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                interval={tickInterval}
                className="text-xs"
                tick={{ fill: "hsl(var(--muted-foreground))" }}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                width={50}
                className="text-xs"
                tick={{ fill: "hsl(var(--muted-foreground))" }}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <ChartTooltip
                cursor={{
                  stroke: "hsl(var(--muted-foreground))",
                  strokeWidth: 1,
                }}
                content={
                  <ChartTooltipContent
                    formatter={(value) => [
                      `${Number(value).toFixed(3)} ${unit}`,
                      "Trend",
                    ]}
                    labelFormatter={(label, payload) => {
                      if (payload?.[0]?.payload?.fullDate) {
                        return payload[0].payload.fullDate;
                      }
                      return label;
                    }}
                  />
                }
              />
              <Line
                type="monotone"
                dataKey="trend"
                stroke="hsl(var(--chart-2))"
                strokeWidth={2.5}
                dot={false}
              />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* Chart 3: Seasonal Component */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[hsl(var(--chart-3))]"></div>
            Seasonal Component
          </CardTitle>
          <CardDescription className="text-xs">
            Repeating patterns and cycles
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[220px] w-full">
            <AreaChart
              data={chartData}
              margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient
                  id={`gradient-seasonal-${param}`}
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop
                    offset="0%"
                    stopColor="hsl(var(--chart-3))"
                    stopOpacity={0.4}
                  />
                  <stop
                    offset="100%"
                    stopColor="hsl(var(--chart-3))"
                    stopOpacity={0.05}
                  />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
                className="stroke-muted"
              />
              <XAxis
                dataKey="date"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                interval={tickInterval}
                className="text-xs"
                tick={{ fill: "hsl(var(--muted-foreground))" }}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                width={50}
                className="text-xs"
                tick={{ fill: "hsl(var(--muted-foreground))" }}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <ChartTooltip
                cursor={{
                  stroke: "hsl(var(--muted-foreground))",
                  strokeWidth: 1,
                }}
                content={
                  <ChartTooltipContent
                    formatter={(value) => [
                      `${Number(value).toFixed(3)} ${unit}`,
                      "Seasonal",
                    ]}
                    labelFormatter={(label, payload) => {
                      if (payload?.[0]?.payload?.fullDate) {
                        return payload[0].payload.fullDate;
                      }
                      return label;
                    }}
                  />
                }
              />
              <Area
                type="monotone"
                dataKey="seasonal"
                stroke="hsl(var(--chart-3))"
                strokeWidth={2}
                fill={`url(#gradient-seasonal-${param})`}
                dot={false}
              />
            </AreaChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* Chart 4: Residual Component */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[hsl(var(--chart-4))]"></div>
            Residual Component
          </CardTitle>
          <CardDescription className="text-xs">
            Random variations and noise
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[220px] w-full">
            <ScatterChart
              data={chartData}
              margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
                className="stroke-muted"
              />
              <XAxis
                dataKey="date"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                interval={tickInterval}
                className="text-xs"
                tick={{ fill: "hsl(var(--muted-foreground))" }}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                width={50}
                className="text-xs"
                tick={{ fill: "hsl(var(--muted-foreground))" }}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <ChartTooltip
                cursor={{
                  stroke: "hsl(var(--muted-foreground))",
                  strokeWidth: 1,
                }}
                content={
                  <ChartTooltipContent
                    formatter={(value) => [
                      `${Number(value).toFixed(3)} ${unit}`,
                      "Residual",
                    ]}
                    labelFormatter={(label, payload) => {
                      if (payload?.[0]?.payload?.fullDate) {
                        return payload[0].payload.fullDate;
                      }
                      return label;
                    }}
                  />
                }
              />
              {/* Zero line reference */}
              <ReferenceLine
                y={0}
                stroke="hsl(var(--muted-foreground))"
                strokeWidth={1}
                strokeDasharray="5 5"
              />
              <Scatter
                dataKey="residual"
                fill="hsl(var(--chart-4))"
                fillOpacity={0.7}
              />
            </ScatterChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  );
}

export function DecompositionChart({
  collectionName,
}: DecompositionChartProps) {
  // Get available parameters
  const { data: availableParams, isLoading: isLoadingParams } = useQuery({
    queryKey: ["decomposition-params", collectionName],
    queryFn: () => GetAvailableDecompositionParameters(collectionName),
    refetchOnWindowFocus: false,
  });

  if (isLoadingParams) {
    return <LoadingSkeleton />;
  }

  if (!availableParams || availableParams.length === 0) {
    return (
      <EmptyState message="Tidak ada data dekomposisi yang tersedia untuk dataset ini" />
    );
  }

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <h2 className="text-2xl font-bold tracking-tight">
          Analisis Dekomposisi
        </h2>
        <p className="text-muted-foreground">
          Visualisasi dekomposisi time series untuk memahami komponen trend,
          seasonal, dan residual
        </p>
      </div>

      <Tabs defaultValue={availableParams[0]} className="w-full">
        <TabsList className="mb-4 flex-wrap h-auto gap-2 bg-transparent p-0">
          {availableParams.map((param) => (
            <TabsTrigger
              key={param}
              value={param}
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-lg px-4 py-2 border"
            >
              {getParamLabel(param)}
            </TabsTrigger>
          ))}
        </TabsList>

        {availableParams.map((param) => (
          <TabsContent key={param} value={param} className="mt-0">
            <ParameterDecompositionView
              param={param}
              collectionName={collectionName}
            />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
