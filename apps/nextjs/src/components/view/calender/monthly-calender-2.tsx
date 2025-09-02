"use client";

import { useQuery } from "@tanstack/react-query";
import { getHoltWinterDaily } from "@/lib/fetch/files.fetch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useMemo } from "react";
import clsx from "clsx";
import { format, parse, eachDayOfInterval, getDay } from "date-fns";
import { id } from "date-fns/locale";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const KT_PERIODS = {
  "KT-1": ["09-20-2025", "01-20-2026"],
  "KT-2": ["01-21-2026", "06-20-2026"],
  "KT-3": ["06-21-2026", "09-19-2026"],
};

// === FUNGSI INI DIUBAH ===
const getWeatherColor = (rain: number, temp: number, humidity: number) => {
  // 1. Cek kesesuaian untuk setiap parameter
  const isRainSesuai = rain >= 5.7 && rain <= 16.7;
  const isTempSesuai = temp >= 24 && temp <= 29;
  const isHumiditySesuai = humidity >= 33 && humidity <= 90;

  // 2. Hitung berapa banyak parameter yang "Sesuai"
  // (Boolean diubah menjadi Angka: true=1, false=0)
  const sesuaiCount = Number(isRainSesuai) + Number(isTempSesuai) + Number(isHumiditySesuai);

  // 3. Tentukan warna berdasarkan aturan "minimal 2 dari 3"
  if (sesuaiCount === 3) {
    return "bg-green-300"; // Sangat Cocok (3/3 parameter terpenuhi)
  }
  if (sesuaiCount === 2) {
    return "bg-green-100"; // Cukup Cocok (2/3 parameter terpenuhi)
  }

  // Jika kurang dari 2 parameter yang sesuai (0 atau 1), maka Tidak Cocok
  return "bg-red-300";
};

const getPeriodCalendarGrid = (data: any[], startDate: Date, endDate: Date) => {
  const days = eachDayOfInterval({ start: startDate, end: endDate });
  const startOffset = getDay(startDate) === 0 ? 6 : getDay(startDate) - 1;

  const grid: (any | null)[] = Array(startOffset).fill(null);
  days.forEach((date) => {
    const forecast = data.find((item) => format(new Date(item.forecast_date), "yyyy-MM-dd") === format(date, "yyyy-MM-dd"));
    if (forecast) grid.push(forecast);
    else grid.push({ forecast_date: date.toISOString(), parameters: { RR_imputed: { forecast_value: 0 } } });
  });

  const rows: (any | null)[][] = [];
  for (let i = 0; i < grid.length; i += 1) {
    const rowIndex = i % 7;
    if (!rows[rowIndex]) rows[rowIndex] = [];
    rows[rowIndex].push(grid[i]);
  }

  return rows;
};

export default function PeriodCalendar() {
  const {
    data: forecastData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["holt-winter-period"],
    queryFn: async () => {
      const all = await getHoltWinterDaily(1, 731);
      return all.items
        .map((item: any) => ({
          ...item,
          forecast_date: new Date(item.forecast_date).toISOString(),
        }))
        .sort((a: any, b: any) => new Date(a.forecast_date).getTime() - new Date(b.forecast_date).getTime());
    },
  });

  const periodRows = useMemo(() => {
    if (!forecastData || forecastData.length === 0) return { "KT-1": [], "KT-2": [], "KT-3": [] };
    const rows: Record<string, (any | null)[][]> = {};
    Object.keys(KT_PERIODS).forEach((period) => {
      const [startStr, endStr] = KT_PERIODS[period as keyof typeof KT_PERIODS];
      const startDate = parse(startStr, "MM-dd-yyyy", new Date());
      const endDate = parse(endStr, "MM-dd-yyyy", new Date());
      rows[period] = getPeriodCalendarGrid(forecastData, startDate, endDate);
    });
    return rows;
  }, [forecastData]);

  const renderPeriodGrid = (period: keyof typeof KT_PERIODS) => (
    <TabsContent value={period}>
      <ScrollArea className="w-full overflow-auto">
        <div className="min-w-[900px]">
          <Table className="bg-background  ">
            <TableHeader>
              <TableRow className="*:border-border">
                {Array.from({ length: periodRows[period][0]?.length || 0 }).map((_, colIdx) => (
                  <TableHead key={colIdx} className="text-center">
                    {periodRows[period][0][colIdx] ? format(new Date(periodRows[period][0][colIdx].forecast_date), " MMM", { locale: id }) : ""}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              <TooltipProvider>
                {["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"].map((day, rowIdx) => (
                  <TableRow key={day} className="*:border-border">
                    <TableCell className="text-center font-medium">{day}</TableCell>

                    {periodRows[period][rowIdx]?.map((dayData, colIdx) => {
                      if (dayData) {
                        return (
                          <Tooltip key={colIdx}>
                            <TooltipTrigger asChild>
                              <TableCell
                                className={clsx(
                                  "text-center relative h-20 w-20 border",
                                  getWeatherColor(dayData.parameters?.RR_imputed?.forecast_value ?? 0, dayData.parameters?.TAVG?.forecast_value ?? 0, dayData.parameters?.RH_AVG_preprocessed?.forecast_value ?? 0)
                                )}
                              >
                                <div className="text-[10px] absolute top-0 right-1">{format(new Date(dayData.forecast_date), "dd")}</div>
                              </TableCell>
                            </TooltipTrigger>
                            <TooltipContent side="top">
                              <div className="text-xs">
                                <p>Hujan: {dayData.parameters?.RR_imputed?.forecast_value.toFixed(2)} mm</p>
                                <p>Suhu: {dayData.parameters?.TAVG?.forecast_value.toFixed(2)} °C</p>
                                <p>Kelembaban: {dayData.parameters?.RH_AVG_preprocessed?.forecast_value.toFixed(2)} %</p>
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        );
                      } else {
                        return (
                          <TableCell key={colIdx} className="text-center h-20 w-20 text-xs">
                            -
                          </TableCell>
                        );
                      }
                    })}
                  </TableRow>
                ))}
              </TooltipProvider>
            </TableBody>
          </Table>
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </TabsContent>
  );

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <div className="text-red-500 p-4 border border-red-200 rounded">Error: {error instanceof Error ? error.message : "Unknown error"}</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="KT-1">
        <TabsList className="mb-4">
          <TabsTrigger value="KT-1">KT-1</TabsTrigger>
          <TabsTrigger value="KT-2">KT-2</TabsTrigger>
          <TabsTrigger value="KT-3">KT-3</TabsTrigger>
        </TabsList>
        {renderPeriodGrid("KT-1")}
        {renderPeriodGrid("KT-2")}
        {renderPeriodGrid("KT-3")}
      </Tabs>
      {/* === BAGIAN INI DIUBAH === */}
      <div className="flex flex-col gap-2 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-6 h-4 bg-green-300 border" />
          <span>Sangat Cocok Tanam (3/3 parameter sesuai)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-6 h-4 bg-green-100 border" />
          <span>Cukup Cocok Tanam (2/3 parameter sesuai)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-6 h-4 bg-red-300 border" />
          <span>Tidak Cocok Tanam (&lt;2 parameter sesuai)</span>
        </div>
      </div>
    </div>
  );
}
