"use client";

import { useQuery } from "@tanstack/react-query";
import { getForecastConfigs } from "@/lib/fetch/files.fetch";
import { LabelList, Pie, PieChart } from "recharts";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { PieChartIcon } from "lucide-react";
import { useMemo } from "react";

const COLORS = {
  mae: "hsl(var(--chart-1))",
  mape: "hsl(var(--chart-3))",
  rmse: "hsl(var(--chart-2))",
};

const chartConfig = {
  mae: { label: "MAE", color: COLORS.mae },
  mape: { label: "MAPE", color: COLORS.mape },
  rmse: { label: "RMSE", color: COLORS.rmse },
} satisfies ChartConfig;

// Mapping nama kolom
const COLUMN_NAME_MAPPING: Record<string, string> = {
  // Kelembapan
  'RH_AVG_preprocessed': 'Kelembaban Rata-rata',
  'RH_AVG': "Kelembaban Rata-rata",

  // Suhu BMKG
  'TN': 'Suhu Minimum',
  'TX': 'Suhu Maksimum',
  'TAVG': 'Suhu Rata-rata',

  // Rainfall
  'RR_original': 'Curah Hujan Asli',
  'RR_imputed': 'Curah Hujan (Diperbaiki)',
  'is_outlier': 'Status Outlier',
  'RR_log': 'Curah Hujan (Log)',
  'RR_sqrt': 'Curah Hujan (Akar)',
  'RR_boxcox': 'Curah Hujan (Box-Cox)',
  'RR': 'Curah Hujan',

  // NASA
  'T2M': 'Suhu Udara',
  'T2M_MAX': 'Suhu Maksimum',
  'T2M_MIN': 'Suhu Minimum',
  'RH2M': 'Kelembaban Udara',
  'PRECTOTCORR': 'Curah Hujan',
  'ALLSKY_SFC_SW_DWN': 'Radiasi Matahari',
  'WS10M': 'Kecepatan Angin',
  'WS10M_MAX': 'Angin Maksimum',
  'WD10M': 'Arah Angin',

  'SS': 'Lamanya Penyinaran Matahari (Jam)',
  'FF_X': 'Kecepatan Angin Maksimum',
  'DDD_X': 'Arah Angin Maksimum',
  'FF_AVG': 'Kecepatan Angin Rata-rata',
  'DDD_CAR': 'Arah Angin Terbanyak',

  // Umum
  'Date': 'Tanggal',
  'Year': 'Tahun',
  'Month': 'Bulan',
  'Day': 'Hari',
  'month': 'Bulan',
  'day': 'Hari',
};

// Mapping collection name ke label singkat
const COLLECTION_LABELS: Record<string, string> = {
  'Nasa Pidie Kec Indrajaya': 'Pidie',
  'Nasa Aceh Besar Kec Darussalam': 'Aceh Besar',
  'Nasa Aceh Utara Kec Lhoksukon': 'Aceh Utara',
  'kelembapan': 'BMKG',
  'suhu bmkg': 'BMKG',
  'rainfall': 'BMKG',
};

// Fungsi helper
const getDisplayName = (col: string): string => {
  if (COLUMN_NAME_MAPPING[col]) {
    return COLUMN_NAME_MAPPING[col];
  }
  return col
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
};

const getCollectionLabel = (collectionName: string): string => {
  return COLLECTION_LABELS[collectionName] || collectionName;
};

export function RoundedPieChart() {
  const { data = [], isLoading } = useQuery({
    queryKey: ["forecast-config"],
    queryFn: getForecastConfigs,
  });

  // Build dynamic mapping dari config
  const { latestConfig, paramToCollection } = useMemo(() => {
    const completed = data.filter((item: any) => item.status === "done" && item.error_metrics?.length > 0);
    const config = completed.length > 0 ? completed[0] : null;
    
    if (!config?.columns) {
      return { latestConfig: null, paramToCollection: {} };
    }
    
    const mapping: Record<string, string> = {};
    config.columns.forEach((col: any) => {
      mapping[col.columnName] = col.collectionName;
    });
    
    return { latestConfig: config, paramToCollection: mapping };
  }, [data]);

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[...Array(2)].map((_, i) => (
          <Card key={i}>
            <CardHeader className="pb-2">
              <Skeleton className="h-5 w-3/4" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-[200px] w-full rounded-lg" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (!latestConfig) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center justify-center py-10">
          <PieChartIcon className="h-10 w-10 text-muted-foreground mb-3" />
          <p className="font-medium text-base">Belum ada data metrik</p>
          <p className="text-sm text-muted-foreground">
            Jalankan peramalan untuk melihat metrik error
          </p>
        </CardContent>
      </Card>
    );
  }

  const errorMetricsArray = latestConfig.error_metrics ?? [];

  return (
    <TooltipProvider>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {errorMetricsArray.map((entry: any, index: number) => {
          const metrics = entry.metrics ?? {};
          // Ambil collection dari mapping, fallback ke entry.collectionName
          const collectionName = paramToCollection[entry.columnName] || entry.collectionName || "Unknown";
          const collectionLabel = getCollectionLabel(collectionName);
          
          const chartData = [
            { key: "mae", value: metrics.mae || 0, fill: COLORS.mae },
            { key: "mape", value: metrics.mape || 0, fill: COLORS.mape },
            { key: "rmse", value: metrics.rmse || 0, fill: COLORS.rmse },
          ];

          const displayName = getDisplayName(entry.columnName);

          return (
            <Card key={index}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <CardTitle className="text-base sm:text-lg font-medium cursor-help">
                        {displayName}
                      </CardTitle>
                    </TooltipTrigger>
                    <TooltipContent side="top">
                      <p className="text-sm">
                        Kolom: <code className="bg-muted px-1 rounded">{entry.columnName}</code>
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </div>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  {collectionLabel}
                </p>
              </CardHeader>

              <CardContent className="pt-0">
                <div className="flex flex-col sm:flex-row items-center gap-4">
                  <ChartContainer config={chartConfig} className="mx-auto aspect-square h-[180px] sm:h-[180px]">
                    <PieChart>
                      <ChartTooltip content={<ChartTooltipContent nameKey="key" hideLabel />} />
                      <Pie 
                        data={chartData} 
                        dataKey="value" 
                        nameKey="key"
                        innerRadius={40} 
                        outerRadius={70}
                        cornerRadius={4} 
                        paddingAngle={2}
                      >
                        <LabelList 
                          dataKey="value" 
                          stroke="none" 
                          fontSize={10} 
                          fontWeight={500}
                          formatter={(value: number) => value.toFixed(1)} 
                        />
                      </Pie>
                    </PieChart>
                  </ChartContainer>

                  <div className="flex flex-col gap-2">
                    {chartData.map((item) => (
                      <div key={item.key} className="flex items-center gap-2">
                        <div 
                          className="h-3 w-3 rounded-sm shrink-0" 
                          style={{ backgroundColor: item.fill }} 
                        />
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground uppercase w-10">
                            {item.key}
                          </span>
                          <span className="text-sm font-medium tabular-nums">
                            {item.value.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </TooltipProvider>
  );
}
