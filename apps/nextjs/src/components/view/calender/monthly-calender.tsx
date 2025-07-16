"use client";

import { useQuery } from "@tanstack/react-query";
import { getHoltWinterDaily } from "@/lib/fetch/files.fetch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useState, useMemo } from "react";
import { format, startOfMonth, endOfMonth, getDay, addDays } from "date-fns";
import { id } from "date-fns/locale";

const getWeatherIcon = (forecastValue: number) => {
  if (forecastValue > 0.5) return "🌧️";
  return "☀️";
};

const getMonthCalendarGrid = (data: any[], year: string, month: string) => {
  const monthStart = startOfMonth(new Date(`${year}-${month}-01`));
  const monthEnd = endOfMonth(monthStart);
  const startOffset = (getDay(monthStart) + 6) % 7; // Mulai dari Senin

  const days: (any | null)[] = Array(startOffset).fill(null);
  for (let d = 0; d <= monthEnd.getDate() - 1; d++) {
    const date = addDays(monthStart, d);
    const forecast = data.find((item) => format(new Date(item.forecast_date), "yyyy-MM-dd") === format(date, "yyyy-MM-dd"));
    if (forecast) days.push(forecast);
    else days.push({ forecast_date: date.toISOString(), parameters: { RR_imputed: { forecast_value: 0 } } });
  }

  // Group into weeks
  const weeks: (any | null)[][] = [];
  for (let i = 0; i < days.length; i += 7) {
    weeks.push(days.slice(i, i + 7));
  }

  return weeks;
};

export default function MonthCalendar() {
  const [year, setYear] = useState("2025");
  const [month, setMonth] = useState("07");

  const {
    data: forecastData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["holt-winter", year, month],
    queryFn: async () => {
      const all = await getHoltWinterDaily(1, 5000);
      return all.items.filter((item: any) => {
        const date = new Date(item.forecast_date);
        return date.getFullYear().toString() === year && ("0" + (date.getMonth() + 1)).slice(-2) === month;
      });
    },
  });

  const weeks = useMemo(() => {
    if (!forecastData || forecastData.length === 0) return [];
    return getMonthCalendarGrid(forecastData, year, month);
  }, [forecastData, year, month]);

  // Loading state
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
      <div className="flex gap-4">
        <Select value={year} onValueChange={setYear}>
          <SelectTrigger className="w-[120px]">
            <SelectValue placeholder="Pilih Tahun" />
          </SelectTrigger>
          <SelectContent>
            {["2025", "2026", "2027"].map((y) => (
              <SelectItem key={y} value={y}>
                {y}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={month} onValueChange={setMonth}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Pilih Bulan" />
          </SelectTrigger>
          <SelectContent>
            {Array.from({ length: 12 }).map((_, i) => {
              const val = ("0" + (i + 1)).slice(-2);
              return (
                <SelectItem key={val} value={val}>
                  {format(new Date(0, i), "MMMM", { locale: id })}
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
      </div>

      <Table className="bg-background mt-6 min-w-[900px]">
        <TableHeader>
          <TableRow className="*:border-border">
            {["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"].map((d) => (
              <TableHead key={d} className="text-center">
                {d}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {weeks.map((week, i) => (
            <TableRow key={i} className="*:border-border">
              {week.map((day, j) => {
                if (!day) return <TableCell key={j} className="text-center" />;
                const val = day.parameters?.RR_imputed?.forecast_value;
                return (
                  <TableCell key={j} className="text-center space-y-1">
                    <div className="text-xs font-semibold">{format(new Date(day.forecast_date), "dd")}</div>
                    <div className="text-xl">{getWeatherIcon(val)}</div>
                    <div className="text-xs text-muted-foreground">{val?.toFixed(2)} mm</div>
                  </TableCell>
                );
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
