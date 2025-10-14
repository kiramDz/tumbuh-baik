"use client";

import { useQuery } from "@tanstack/react-query";
import { getHoltWinterDaily } from "@/lib/fetch/files.fetch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useMemo } from "react";
import clsx from "clsx";
import { format, eachDayOfInterval, getDay, addMonths, subDays, addDays } from "date-fns";
import { id } from "date-fns/locale";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// const KT_PERIODS = {
//   "Periode-1": ["09-20-2025", "01-20-2026"],
//   "Periode-2": ["01-21-2026", "06-20-2026"],
//   "Periode-3": ["06-21-2026", "09-19-2026"],
// };

const getWeatherColor = (rain: number, temp: number, humidity: number) => {
  const isRainSesuai = rain >= 5.7 && rain <= 16.7;
  const isTempSesuai = temp >= 24 && temp <= 29;
  const isHumiditySesuai = humidity >= 33 && humidity <= 90;

  const sesuaiCount = Number(isRainSesuai) + Number(isTempSesuai) + Number(isHumiditySesuai);

  if (sesuaiCount === 3) {
    return "bg-green-300";
  }
  if (sesuaiCount === 2) {
    return "bg-green-100";
  }

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
      const all = await getHoltWinterDaily(1, 365);
      return all.items
        .map((item: any) => ({
          ...item,
          forecast_date: new Date(item.forecast_date).toISOString(),
        }))
        .sort((a: any, b: any) => new Date(a.forecast_date).getTime() - new Date(b.forecast_date).getTime());
    },
  });

  const { periodRows, periodRanges } = useMemo(() => {
    const emptyState = {
      periodRows: { "Periode-1": [], "Periode-2": [], "Periode-3": [] },
      periodRanges: { "Periode-1": "Memuat...", "Periode-2": "Memuat...", "Periode-3": "Memuat..." },
    };
    if (!forecastData || forecastData.length === 0) return emptyState;

    // 1. Tentukan tanggal awal dari data pertama
    const globalStartDate = new Date(forecastData[0].forecast_date);

    // 2. Hitung periode secara dinamis (masing-masing 4 bulan)
    const kt1_endDate = subDays(addMonths(globalStartDate, 4), 1);

    const kt2_startDate = addDays(kt1_endDate, 1);
    const kt2_endDate = subDays(addMonths(kt2_startDate, 4), 1);

    const kt3_startDate = addDays(kt2_endDate, 1);
    const kt3_endDate = subDays(addMonths(kt3_startDate, 4), 1);

    const dynamicPeriods = {
      "Periode-1": { start: globalStartDate, end: kt1_endDate },
      "Periode-2": { start: kt2_startDate, end: kt2_endDate },
      "Periode-3": { start: kt3_startDate, end: kt3_endDate },
    };

    const newPeriodRows: Record<string, (any | null)[][]> = {};
    const newPeriodRanges: Record<string, string> = {};

    Object.keys(dynamicPeriods).forEach((period) => {
      const { start, end } = dynamicPeriods[period as keyof typeof dynamicPeriods];

      // Buat grid kalender
      newPeriodRows[period] = getPeriodCalendarGrid(forecastData, start, end);

      // Buat teks rentang tanggal untuk ditampilkan di UI
      newPeriodRanges[period] = `${format(start, "d MMM yyyy", { locale: id })} - ${format(end, "d MMM yyyy", { locale: id })}`;
    });

    return { periodRows: newPeriodRows, periodRanges: newPeriodRanges };
  }, [forecastData]);

  const renderPeriodGrid = (period: keyof typeof periodRows) => (
    <TabsContent value={period}>
      <div className="mb-3 text-sm font-medium text-center text-muted-foreground">Periode Tanam: {periodRanges[period]}</div>
      <ScrollArea className="w-full overflow-auto">
        <div className="min-w-[900px]">
          <Table className="bg-background  ">
            <TableHeader>
              <TableRow className="*:border-border">
                <TableHead className="w-[100px]">Hari</TableHead>
                {periodRows[period][0]?.map((_, colIdx) => (
                  <TableHead key={colIdx} className="text-center">
                    {`Minggu ${colIdx + 1}`}
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
                      if (dayData && !dayData.isPlaceholder) {
                        return (
                          <Tooltip key={colIdx}>
                            <TooltipTrigger asChild>
                              <TableCell
                                className={clsx(
                                  "text-center relative h-20 w-20 border",
                                  getWeatherColor(dayData.parameters?.RR_imputed?.forecast_value ?? 0, dayData.parameters?.TAVG?.forecast_value ?? 0, dayData.parameters?.RH_AVG_preprocessed?.forecast_value ?? 0)
                                )}
                              >
                                <div className="text-xs absolute top-1 right-1.5">{format(new Date(dayData.forecast_date), "dd")}</div>
                                <div className="text-[10px] absolute bottom-1 left-1.5">{format(new Date(dayData.forecast_date), "MMM", { locale: id })}</div>
                              </TableCell>
                            </TooltipTrigger>
                            <TooltipContent side="top">
                              <div className="text-xs">
                                <p>{format(new Date(dayData.forecast_date), "eeee, d MMMM yyyy", { locale: id })}</p>
                                <p>Hujan: {dayData.parameters?.RR_imputed?.forecast_value.toFixed(2)} mm</p>
                                <p>Suhu: {dayData.parameters?.TAVG?.forecast_value.toFixed(2)} °C</p>
                                <p>Kelembaban: {dayData.parameters?.RH_AVG_preprocessed?.forecast_value.toFixed(2)} %</p>
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        );
                      } else {
                        return <TableCell key={colIdx} className="text-center h-20 w-20 border bg-slate-50"></TableCell>;
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
      <Tabs defaultValue="Periode-1">
        <TabsList className="mb-4">
          <TabsTrigger value="Periode-1">Periode-1</TabsTrigger>
          <TabsTrigger value="Periode-2">Periode-2</TabsTrigger>
          <TabsTrigger value="Periode-3">Periode-3</TabsTrigger>
        </TabsList>
        {renderPeriodGrid("Periode-1")}
        {renderPeriodGrid("Periode-2")}
        {renderPeriodGrid("Periode-3")}
      </Tabs>
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
