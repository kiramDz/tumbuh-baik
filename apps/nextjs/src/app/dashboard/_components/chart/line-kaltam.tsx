"use client";

import { useQuery } from "@tanstack/react-query";
import { CartesianGrid, Line, LineChart, XAxis, YAxis, Area, AreaChart } from "recharts";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { TrendingDown, TrendingUp, Minus, Calendar, Activity } from "lucide-react";
import { getHoltWinterDaily, fetchHistoricalData, getForecastConfigs } from "@/lib/fetch/files.fetch";
import { useState, useMemo } from "react";

const chartConfig = {
  value: {
    label: "Nilai",
    color: "hsl(var(--chart-1))",
  },
  historical: {
    label: "Historis",
    color: "hsl(var(--chart-2))",
  },
} satisfies ChartConfig;

// Mapping nama parameter ke label yang lebih readable
const paramLabels: Record<string, string> = {
  "ALLSKY_SFC_SW_DWN": "Radiasi Matahari",
  "RH_AVG_preprocessed": "Kelembaban Udara",
  "RH_AVG": "Kelembaban Udara",
  "RH2M": "Kelembaban Udara",
  "TAVG": "Suhu Rata-rata",
  "TMAX": "Suhu Maksimum",
  "TMIN": "Suhu Minimum",
  "TX": "Suhu Maksimum",
  "TN": "Suhu Minimum",
  "T2M": "Suhu Udara",
  "RR_imputed": "Curah Hujan",
  "RR": "Curah Hujan",
  "PRECTOTCORR": "Curah Hujan",
  "NDVI": "Indeks Vegetasi",
  "NDVI_imputed": "Indeks Vegetasi",
  "WS2M": "Kecepatan Angin",
  "PS": "Tekanan Udara",
};

// Mapping unit untuk setiap parameter
const paramUnits: Record<string, string> = {
  "ALLSKY_SFC_SW_DWN": "MJ/m²",
  "RH_AVG_preprocessed": "%",
  "RH_AVG": "%",
  "RH2M": "%",
  "TAVG": "°C",
  "TMAX": "°C",
  "TMIN": "°C",
  "TX": "°C",
  "TN": "°C",
  "T2M": "°C",
  "RR_imputed": "mm",
  "RR": "mm",
  "PRECTOTCORR": "mm",
  "NDVI": "",
  "NDVI_imputed": "",
  "WS2M": "m/s",
  "PS": "kPa",
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
          <Skeleton key={i} className="h-9 w-24 rounded-lg" />
        ))}
      </div>
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-32" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[300px] w-full rounded-lg" />
        </CardContent>
      </Card>
    </div>
  );
}

function EmptyState() {
  return (
    <Card className="border-dashed">
      <CardContent className="flex flex-col items-center justify-center py-16">
        <Activity className="h-12 w-12 text-muted-foreground mb-4" />
        <p className="font-medium text-lg">Belum ada data peramalan</p>
        <p className="text-sm text-muted-foreground text-center max-w-sm mt-1">
          Jalankan model peramalan terlebih dahulu untuk melihat grafik prediksi
        </p>
      </CardContent>
    </Card>
  );
}

interface ChartData {
  date: string;
  fullDate: string;
  value: number;
  isHistorical?: boolean;
  year: number;
}

interface ParamChartProps {
  param: string;
  data: ChartData[];
  mode: "forecast-only" | "combined";
}

function ParamChart({ param, data, mode }: ParamChartProps) {
  if (data.length === 0) return null;

  const forecastData = data.filter(d => !d.isHistorical);
  const historicalData = data.filter(d => d.isHistorical);

  const displayData = mode === "forecast-only" ? forecastData : data;

  const firstValue = displayData[0].value;
  const lastValue = displayData[displayData.length - 1].value;
  const percentChange = firstValue !== 0 ? ((lastValue - firstValue) / firstValue) * 100 : 0;
  
  const minValue = Math.min(...displayData.map(d => d.value));
  const maxValue = Math.max(...displayData.map(d => d.value));
  const avgValue = displayData.reduce((sum, d) => sum + d.value, 0) / displayData.length;

  const isUp = percentChange > 1;
  const isDown = percentChange < -1;
  const TrendIcon = isDown ? TrendingDown : isUp ? TrendingUp : Minus;
  
  const trendVariant = isDown ? "destructive" : isUp ? "default" : "secondary";
  const unit = getParamUnit(param);

  const years = [...new Set(displayData.map(d => d.year))].sort();
  const yearRange = years.length;
  
  let tickInterval: number;
  if (mode === "combined") {
    if (yearRange > 20) {
      tickInterval = Math.ceil(displayData.length / 10);
    } else if (yearRange > 10) {
      tickInterval = Math.ceil(displayData.length / 15);
    } else {
      tickInterval = Math.ceil(displayData.length / 20);
    }
  } else {
    tickInterval = Math.ceil(displayData.length / 12);
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="text-xl font-semibold">
              {getParamLabel(param)}
            </CardTitle>
            <CardDescription className="flex items-center gap-2">
              <Calendar className="h-3.5 w-3.5" />
              {mode === "combined" 
                ? `Data historis (${years[0]}-${years[years.length - 1]}) & prediksi` 
                : `Prediksi 365 hari ke depan`
              }
            </CardDescription>
          </div>
          <Badge variant={trendVariant} className="flex items-center gap-1">
            <TrendIcon className="h-3.5 w-3.5" />
            {percentChange >= 0 ? "+" : ""}{percentChange.toFixed(1)}%
          </Badge>
        </div>
        
        <div className="grid grid-cols-3 gap-4 pt-4">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Minimum</p>
            <p className="text-sm font-medium tabular-nums">
              {minValue.toFixed(2)} {unit}
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Rata-rata</p>
            <p className="text-sm font-medium tabular-nums">
              {avgValue.toFixed(2)} {unit}
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Maksimum</p>
            <p className="text-sm font-medium tabular-nums">
              {maxValue.toFixed(2)} {unit}
            </p>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-4">
        <ChartContainer config={chartConfig} className="h-[280px] w-full">
          <AreaChart 
            data={displayData} 
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id={`gradient-forecast-${param}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(var(--chart-1))" stopOpacity={0.3} />
                <stop offset="100%" stopColor="hsl(var(--chart-1))" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id={`gradient-historical-${param}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(var(--chart-2))" stopOpacity={0.3} />
                <stop offset="100%" stopColor="hsl(var(--chart-2))" stopOpacity={0.05} />
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
              tick={{ fill: 'hsl(var(--muted-foreground))' }}
            />
            <YAxis 
              tickLine={false} 
              axisLine={false}
              tickMargin={8}
              width={50}
              className="text-xs"
              tick={{ fill: 'hsl(var(--muted-foreground))' }}
              tickFormatter={(value) => value.toFixed(1)}
            />
            <ChartTooltip 
              cursor={{ stroke: 'hsl(var(--muted-foreground))', strokeWidth: 1 }}
              content={
                <ChartTooltipContent 
                  formatter={(value, name, item) => {
                    const label = item.payload.isHistorical ? "Historis" : "Prediksi";
                    return [`${Number(value).toFixed(2)} ${unit}`, label];
                  }}
                  labelFormatter={(label, payload) => {
                    if (payload?.[0]?.payload?.fullDate) {
                      return payload[0].payload.fullDate;
                    }
                    return label;
                  }}
                />
              } 
            />
            {mode === "combined" && historicalData.length > 0 && (
              <Area
                type="monotone"
                dataKey={(item) => item.isHistorical ? item.value : null}
                stroke="hsl(var(--chart-2))"
                strokeWidth={2}
                fill={`url(#gradient-historical-${param})`}
                dot={false}
                activeDot={{ r: 4, fill: 'hsl(var(--chart-2))' }}
                connectNulls={false}
              />
            )}
            <Area
              type="monotone"
              dataKey={(item) => mode === "combined" ? (!item.isHistorical ? item.value : null) : item.value}
              stroke="hsl(var(--chart-1))"
              strokeWidth={2}
              fill={`url(#gradient-forecast-${param})`}
              dot={false}
              activeDot={{ r: 4, fill: 'hsl(var(--chart-1))' }}
              connectNulls={mode === "forecast-only"}
            />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

export function RainbowGlowGradientLineChart() {
  const [viewMode, setViewMode] = useState<"forecast-only" | "combined">("forecast-only");

  const { data: rawData, isLoading } = useQuery({
    queryKey: ["hw-daily-full"],
    queryFn: () => getHoltWinterDaily(1, 365),
    refetchOnWindowFocus: false,
  });

  // Fetch forecast config untuk mendapatkan mapping parameter -> collection
  const { data: forecastConfigs } = useQuery({
    queryKey: ["forecast-configs"],
    queryFn: getForecastConfigs,
    refetchOnWindowFocus: false,
  });

  // Build dynamic mapping dari config
  const paramToCollection = useMemo(() => {
    if (!forecastConfigs || forecastConfigs.length === 0) return {};
    
    // Ambil config terakhir yang statusnya "done"
    const latestConfig = forecastConfigs.find((config: any) => config.status === "done") || forecastConfigs[0];
    
    if (!latestConfig?.columns) return {};

    const mapping: Record<string, { collectionName: string; columnName: string }> = {};
    
    latestConfig.columns.forEach((col: any) => {
      mapping[col.columnName] = {
        collectionName: col.collectionName,
        columnName: col.columnName
      };
    });

    console.log("📍 Dynamic parameter mapping:", mapping);
    return mapping;
  }, [forecastConfigs]);

  const { data: historicalDataMap, isLoading: isLoadingHistorical } = useQuery({
    queryKey: ["historical-hw-data", viewMode, paramToCollection],
    queryFn: async () => {
      if (viewMode !== "combined") return {};

      const items = rawData?.items || [];
      if (items.length === 0) return {};

      const parameters = new Set<string>();
      items.forEach((item: any) => {
        Object.keys(item.parameters || {}).forEach((param) => parameters.add(param));
      });

      const paramArray = Array.from(parameters);
      const historicalMap: Record<string, any[]> = {};

      console.log("🔍 Fetching historical data for parameters:", paramArray);
      console.log("🗺️ Using mapping:", paramToCollection);

      await Promise.all(
        paramArray.map(async (param) => {
          const mapping = paramToCollection[param];
          if (mapping) {
            console.log(`📥 Fetching ${param} from ${mapping.collectionName}`);
            const data = await fetchHistoricalData(
              mapping.collectionName, 
              mapping.columnName
            );
            historicalMap[param] = data;
            console.log(`✅ Got ${data.length} records for ${param}`);
          } else {
            console.warn(`⚠️ No mapping found for parameter: ${param}`);
          }
        })
      );

      return historicalMap;
    },
    enabled: viewMode === "combined" && !!rawData && Object.keys(paramToCollection).length > 0,
    refetchOnWindowFocus: false,
  });

  const loading = isLoading || (viewMode === "combined" && isLoadingHistorical);

  if (loading) return <LoadingSkeleton />;

  const items = rawData?.items || [];
  if (items.length === 0) return <EmptyState />;

  // Extract parameter unik
  const parameters = new Set<string>();
  items.forEach((item: any) => {
    Object.keys(item.parameters || {}).forEach((param) => parameters.add(param));
  });

  const paramArray = Array.from(parameters);

  // Group data per parameter, sort by date ascending
  const groupedData: Record<string, ChartData[]> = {};
  paramArray.forEach((param) => {
    const forecastData = items
      .filter((item: any) => item.parameters?.[param]?.forecast_value != null)
      .map((item: any) => {
        const dateObj = new Date(item.forecast_date);
        return {
          date: dateObj.getFullYear().toString(),
          fullDate: dateObj.toLocaleDateString("id-ID", { 
            weekday: "long", 
            year: "numeric", 
            month: "long", 
            day: "numeric" 
          }),
          value: item.parameters[param].forecast_value,
          isHistorical: false,
          year: dateObj.getFullYear(),
        };
      });

    if (viewMode === "combined" && historicalDataMap?.[param]) {
      const historical = historicalDataMap[param].map((item: any) => {
        const dateObj = new Date(item.date);
        return {
          date: dateObj.getFullYear().toString(),
          fullDate: dateObj.toLocaleDateString("id-ID", { 
            weekday: "long", 
            year: "numeric", 
            month: "long", 
            day: "numeric" 
          }),
          value: item.value,
          isHistorical: true,
          year: dateObj.getFullYear(),
        };
      });
      groupedData[param] = [...historical, ...forecastData];
    } else {
      groupedData[param] = forecastData;
    }

    groupedData[param].sort((a: any, b: any) => {
      return new Date(a.fullDate).getTime() - new Date(b.fullDate).getTime();
    });
  });

  if (paramArray.length === 0) return <EmptyState />;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <div className="space-y-2 flex-1 max-w-xs">
          <Label htmlFor="view-mode">Mode Tampilan</Label>
          <Select value={viewMode} onValueChange={(value: any) => setViewMode(value)}>
            <SelectTrigger id="view-mode">
              <SelectValue placeholder="Pilih mode tampilan" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="forecast-only">Hanya Peramalan</SelectItem>
              <SelectItem value="combined">Historis + Peramalan</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <Tabs defaultValue={paramArray[0]} className="w-full">
        <TabsList className="mb-4 flex-wrap h-auto gap-2 bg-transparent p-0">
          {paramArray.map((param) => (
            <TabsTrigger 
              key={param} 
              value={param}
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-lg px-4 py-2 border"
            >
              {getParamLabel(param)}
            </TabsTrigger>
          ))}
        </TabsList>

        {paramArray.map((param) => (
          <TabsContent key={param} value={param} className="mt-0">
            <ParamChart 
              param={param} 
              data={groupedData[param] || []} 
              mode={viewMode} 
            />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
