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

const getSuitability = (rain: number, temp: number, humidity: number, radiation: number) => {
  const criteria = {
    isRainSesuai: rain >= 5.7 && rain <= 16.7,
    isTempSesuai: temp >= 24 && temp <= 29,
    isHumiditySesuai: humidity >= 33 && humidity <= 90,
    isRadiationSesuai: radiation >= 13,
  };

  const sesuaiCount = Object.values(criteria).filter(Boolean).length;

  if (sesuaiCount === 4) {
    return {
      color: "bg-green-300",
      label: "Sangat Cocok",
      count: 4,
    };
  }
  if (sesuaiCount === 3) {
    return {
      color: "bg-green-100",
      label: "Cukup Cocok",
      count: 3,
    };
  }
  return {
    color: "bg-red-300",
    label: "Tidak Cocok",
    count: sesuaiCount,
  };
};

const getPeriodCalendarGrid = (data: any[], startDate: Date, endDate: Date) => {
  const days = eachDayOfInterval({ start: startDate, end: endDate });
  // Offset 0=Senin, 6=Minggu
  const startOffset = getDay(startDate) === 0 ? 6 : getDay(startDate) - 1;

  const grid: (any | null)[] = Array(startOffset).fill(null);
  days.forEach((date) => {
    const forecast = data.find((item) => format(new Date(item.forecast_date), "yyyy-MM-dd") === format(date, "yyyy-MM-dd"));
    if (forecast) grid.push(forecast);
    else
      grid.push({
        forecast_date: date.toISOString(),
        isPlaceholder: true,
        parameters: {},
      });
  });

  const rows: (any | null)[][] = [];
  for (let i = 0; i < grid.length; i += 1) {
    const rowIndex = i % 7; // 0=Senin, 1=Selasa, ...
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

    const globalStartDate = new Date(forecastData[0].forecast_date);
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
      newPeriodRows[period] = getPeriodCalendarGrid(forecastData, start, end);
      newPeriodRanges[period] = `${format(start, "d MMM yyyy", { locale: id })} - ${format(end, "d MMM yyyy", { locale: id })}`;
    });

    return { periodRows: newPeriodRows, periodRanges: newPeriodRanges };
  }, [forecastData]);

  const renderPeriodGrid = (period: keyof typeof periodRows) => (
    <TabsContent value={period}>
      <div className="mb-3 text-sm font-medium text-center text-muted-foreground">Periode Tanam: {periodRanges[period]}</div>
      <ScrollArea className="w-full overflow-auto">
        <div className="min-w-[900px]">
          <Table className="bg-background">
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
                        const rain = dayData.parameters?.RR_imputed?.forecast_value ?? 0;
                        const temp = dayData.parameters?.TAVG?.forecast_value ?? 0;
                        const humid = dayData.parameters?.RH_AVG_preprocessed?.forecast_value ?? 0;

                        const radiation = dayData.parameters?.ALLSKY_SFC_SW_DWN?.forecast_value ?? 0;

                        const suitability = getSuitability(rain, temp, humid, radiation);
                        // --- Akhir Logika Diperbarui ---

                        return (
                          <Tooltip key={colIdx}>
                            <TooltipTrigger asChild>
                              <TableCell
                                className={clsx(
                                  "text-center relative h-20 w-20 border",
                                  suitability.color // Gunakan warna dari objek
                                )}
                              >
                                <div className="text-xs absolute top-1 right-1.5">{format(new Date(dayData.forecast_date), "dd")}</div>
                                <div className="text-[10px] absolute bottom-1 left-1.5">{format(new Date(dayData.forecast_date), "MMM", { locale: id })}</div>
                              </TableCell>
                            </TooltipTrigger>
                            <TooltipContent side="top">
                              {/* --- Tooltip Diperbarui Disini --- */}
                              <div className="text-xs space-y-0.5">
                                <p>{format(new Date(dayData.forecast_date), "eeee, d MMMM yyyy", { locale: id })}</p>
                                <p>Hujan: {rain.toFixed(2)} mm</p>
                                <p>Suhu: {temp.toFixed(2)} °C</p>
                                <p>Kelembaban: {humid.toFixed(2)} %</p>
                                <p>Radiasi: {radiation.toFixed(2)} W/m²</p>
                                <p className="font-bold pt-1">
                                  {suitability.label} ({suitability.count}/4)
                                </p>
                              </div>
                              {/* --- Akhir Tooltip Diperbarui --- */}
                            </TooltipContent>
                          </Tooltip>
                        );
                      } else {
                        // Ini adalah sel kosong (sebelum tanggal mulai) atau data hilang
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

      {/* --- Legend Diperbarui Disini --- */}
      <div className="flex flex-col gap-2 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-6 h-4 bg-green-300 border" />
          <span>Sangat Cocok Tanam (4/4 parameter sesuai)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-6 h-4 bg-green-100 border" />
          <span>Cukup Cocok Tanam (3/4 parameter sesuai)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-6 h-4 bg-red-300 border" />
          <span>Tidak Cocok Tanam (&lt;3 parameter sesuai)</span>
        </div>
      </div>
    </div>
  );
}
