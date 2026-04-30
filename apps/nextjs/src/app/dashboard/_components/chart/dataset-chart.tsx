"use client";

import { GetChartDataBySlug } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  CartesianGrid,
  Line,
  ComposedChart,
  XAxis,
  YAxis,
  Scatter,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Icons } from "@/app/dashboard/_components/icons";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import WindRoseChart from "./wind-rose-chart";

interface ChartSectionProps {
  collectionName: string;
}

const paramLabels: Record<string, string> = {
  // NASA Variable
  T2M: "Suhu Udara (2m)",
  T2M_MAX: "Suhu Maksimum (2m)",
  T2M_MIN: "Suhu Minimum (2m)",
  RH2M: "Kelembaban Relatif (2m)",
  PRECTOTCORR: "Curah Hujan",
  ALLSKY_SFC_SW_DWN: "Radiasi Matahari",
  WS10M: "Kecepatan Angin (10m)",
  WS10M_MAX: "Kecepatan Angin Maksimum (10m)",
  WD10M: "Arah Angin (10m)",

  // BMKG Variable
  TN: "Temperatur Minimum",
  TX: "Temperatur Maksimum",
  TAVG: "Temperatur Rata-rata",
  RH_AVG: "Kelembapan Rata-rata",
  RR: "Curah Hujan",
  SS: "Lamanya Penyinaran Matahari",
  FF_X: "Kecepatan Angin Maksimum",
  DDD_X: "Arah Angin saat Kecepatan Maksimum",
  FF_AVG: "Kecepatan Angin Rata-rata",
  DDD_CAR: "Arah Angin Terbanyak",
};

const paramUnits: Record<string, string> = {
  // NASA
  T2M: "°C",
  T2M_MAX: "°C",
  T2M_MIN: "°C",
  RH2M: "%",
  PRECTOTCORR: "mm/hari",
  ALLSKY_SFC_SW_DWN: "MJ/m²/hari",
  WS10M: "m/s",
  WS10M_MAX: "m/s",
  WD10M: "derajat",

  // BMKG
  TN: "°C",
  TX: "°C",
  TAVG: "°C",
  RH_AVG: "%",
  RR: "mm",
  SS: "jam",
  FF_X: "m/s",
  DDD_X: "derajat",
  FF_AVG: "m/s",
  DDD_CAR: "°",
};

function getParamLabel(param: string): string {
  return paramLabels[param] || param;
}

function getParamUnit(param: string): string {
  return paramUnits[param] || "";
}

// Helper function to check if value is invalid
function isInvalidValue(value: number, columnName?: string): boolean {
  if (Number.isNaN(value)) return true;
  if (value === 8888 || value === 9999) return true;

  // Treat 0 as invalid except for rainfall metrics
  if (value === 0 && columnName !== "RR" && columnName !== "PRECTOTCORR") {
    return true;
  }

  return false;
}

export default function ChartSection({ collectionName }: ChartSectionProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["chart-data", collectionName],
    queryFn: () => GetChartDataBySlug(collectionName),
  });

  const [selectedColumn, setSelectedColumn] = useState<string>("");

  // Filter out unwanted date/time parts from numeric columns
  const filteredNumericColumns =
    data?.numericColumns?.filter(
      (col: string) =>
        !["year", "month", "day", "tahun", "bulan", "hari"].includes(
          col.toLowerCase(),
        ),
    ) || [];

  if (data && filteredNumericColumns.length > 0 && !selectedColumn) {
    setSelectedColumn(filteredNumericColumns[0]);
  }

  if (isLoading) {
    return (
      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Icons.spinner className="h-5 w-5 animate-spin" />
            Memuat Chart...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center">
            <div className="text-muted-foreground">
              Sedang memuat data chart...
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className="mt-6">
        <Icons.alert className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          Gagal memuat data chart. Silakan coba lagi.
        </AlertDescription>
      </Alert>
    );
  }

  if (!data || !data.items?.length || !data.dateColumn) {
    return (
      <Alert className="mt-6 border-yellow-500">
        <Icons.alert className="h-4 w-4 text-yellow-600" />
        <AlertTitle className="text-yellow-600">
          Data Tidak Dapat Ditampilkan
        </AlertTitle>
        <AlertDescription>
          Dataset ini tidak memiliki kolom <strong>Date</strong> atau tidak ada
          data yang bisa ditampilkan.
        </AlertDescription>
      </Alert>
    );
  }

  if (!filteredNumericColumns.length) {
    return (
      <Alert className="mt-6 border-yellow-500">
        <Icons.alert className="h-4 w-4 text-yellow-600" />
        <AlertTitle className="text-yellow-600">
          Data Tidak Dapat Ditampilkan
        </AlertTitle>
        <AlertDescription>
          Dataset ini tidak memiliki kolom numerik yang bisa ditampilkan.
        </AlertDescription>
      </Alert>
    );
  }

  // Pre-calculate minimum valid value to anchor invalid markers at the bottom
  const validValues = data.items
    .map((item: any) => {
      const v = item[selectedColumn];
      return (v === null || v === undefined || v === "") ? NaN : Number(v);
    })
    .filter((val: number) => !isInvalidValue(val, selectedColumn));

  // Substract 3 to match the Y-axis bottom margin
  const bottomMarkerY =
    validValues.length > 0 ? Math.min(...validValues) - 3 : 0;

  const chartData = data.items.map((item: any) => {
    const rawValue = item[selectedColumn];
    const raw = (rawValue === null || rawValue === undefined || rawValue === "") ? NaN : Number(rawValue);
    const isInvalid = isInvalidValue(raw, selectedColumn);

    return {
      date: item[data.dateColumn],
      value: isInvalid ? null : raw,
      invalidValue: isInvalid ? bottomMarkerY : null, // Set to bottom Y-coordinate
      rawInvalidValue: isInvalid ? (Number.isNaN(raw) ? "NaN" : raw) : null, // Preserve real value for tooltip
    };
  });
  // Count statistics
  const totalCount = chartData.length;
  const validCount = chartData.filter((d) => d.value !== null).length;
  const invalidCount = chartData.filter((d) => d.invalidValue !== null).length;

  const chartConfig = {
    value: {
      label: getParamLabel(selectedColumn),
      color: "hsl(var(--chart-1))",
    },
    invalid: {
      label: "Data Invalid",
      color: "hsl(var(--destructive))",
    },
  } satisfies ChartConfig;

  // Determine if this is a Wind Direction parameter
  const isWindDirection = ["WD10M", "DDD_CAR", "DDD_X"].includes(
    selectedColumn,
  );

  // Find the paired speed column
  let pairedSpeedColumn = "";
  if (selectedColumn === "WD10M") pairedSpeedColumn = "WS10M";
  if (selectedColumn === "DDD_CAR" || selectedColumn === "DDD_X")
    pairedSpeedColumn = "FF_AVG";

  return (
    <Card className="mt-6">
      <CardHeader>
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Icons.trendingUp className="h-5 w-5" />
              {getParamLabel(selectedColumn)}
            </CardTitle>
            <div className="flex gap-2 pt-2 flex-wrap">
              <Badge variant="outline" className="text-xs">
                Parameter: {selectedColumn}
              </Badge>
              {getParamUnit(selectedColumn) && (
                <Badge variant="outline" className="text-xs">
                  Unit: {getParamUnit(selectedColumn)}
                </Badge>
              )}
              <Badge variant="outline" className="text-xs bg-blue-50">
                {validCount} data valid
              </Badge>
              {invalidCount > 0 && (
                <Badge variant="destructive" className="text-xs">
                  {invalidCount} data invalid
                </Badge>
              )}
            </div>
          </div>
          <Select value={selectedColumn} onValueChange={setSelectedColumn}>
            <SelectTrigger className="w-full sm:w-[250px]">
              <SelectValue placeholder="Pilih kolom untuk chart" />
            </SelectTrigger>
            <SelectContent>
              {filteredNumericColumns.map((column: string) => (
                <SelectItem key={column} value={column}>
                  <div className="flex items-center justify-between gap-2">
                    <span>{getParamLabel(column)}</span>
                    <span className="text-xs text-muted-foreground">
                      ({column})
                    </span>
                  </div>
                </SelectItem>
              ))}
              {/* If BMKG has DDD_CAR as string, it might not be in numericColumns, add logic here if needed */}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>

      <CardContent>
        {isWindDirection && pairedSpeedColumn ? (
          <WindRoseChart
            data={data.items}
            directionColumn={selectedColumn}
            speedColumn={pairedSpeedColumn}
          />
        ) : (
          <ChartContainer config={chartConfig}>
            <ComposedChart
              accessibilityLayer
              data={chartData}
              margin={{ left: 12, right: 12 }}
            >
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
              <YAxis
                domain={["dataMin - 3", "dataMax + 3"]}
                tickFormatter={(value) => {
                  const unit = getParamUnit(selectedColumn);
                  return unit ? `${value.toFixed(1)}${unit}` : value.toFixed(1);
                }}
              />

              <ChartTooltip
                cursor={false}
                content={
                  <ChartTooltipContent
                    hideLabel={false}
                    labelFormatter={(label) => {
                      if (typeof label === "string" && label.includes("T")) {
                        return label.split("T")[0];
                      }
                      if (label instanceof Date) {
                        return label.toISOString().split("T")[0];
                      }
                      return label;
                    }}
                    formatter={(value, _name, props) => {
                      // Extract payload to check if this is an invalid point
                      const payload = props?.payload;
                      const isInvalidPoint =
                        payload?.rawInvalidValue !== null &&
                        payload?.rawInvalidValue !== undefined;

                      // Use raw invalid value if it's a scatter point, otherwise use normal value
                      const displayValue = isInvalidPoint
                        ? payload.rawInvalidValue
                        : value;
                      
                      const formattedValue = isInvalidPoint
                        ? displayValue
                        : Number(displayValue).toFixed(2);

                      return [
                        `${formattedValue} ${getParamUnit(selectedColumn)}`,
                        isInvalidPoint ? "Data Invalid" : "",
                      ];
                    }}
                  />
                }
              />

              {/* Time series line */}
              <Line
                dataKey="value"
                type="linear"
                stroke="#2563eb"
                dot={false}
                strokeWidth={2}
                connectNulls
              />
              {/* invalid marker */}
              <Scatter dataKey="invalidValue" fill="#ef4444" shape="circle" />
            </ComposedChart>
          </ChartContainer>
        )}
      </CardContent>
    </Card>
  );
}
