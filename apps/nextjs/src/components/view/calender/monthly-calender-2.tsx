"use client";

import React, { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getHoltWinterDaily, getLSTMDaily } from "@/lib/fetch/files.fetch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import clsx from "clsx";
import { format, eachDayOfInterval, getDay, addMonths, subDays, addDays, startOfMonth, getDaysInMonth } from "date-fns";
import { id } from "date-fns/locale";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar, Droplets, Thermometer, Wind, Sun, Info, Loader2, XCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";

type ModelType = "holt-winters" | "lstm";

// ============================================================
// HELPER FUNCTIONS
// ============================================================

const getParameterValue = (parameters: any, paramNames: string[]): number => {
  if (!parameters) return 0;
  
  for (const name of paramNames) {
    const value = parameters?.[name]?.forecast_value;
    if (value !== undefined && value !== null && !isNaN(value)) {
      return value;
    }
  }
  return 0;
};

// --- [LOGIKA PENILAIAN LENGKAP] ---
const getSuitability = (rain: number, temp: number, humidity: number, radiation: number) => {
  
  const criteria = {
    isRainSesuai: rain >= 5.7 && rain <= 16.7, // Syarat Mutlak Tanam
    isTempSesuai: temp >= 24 && temp <= 29,
    isHumiditySesuai: humidity >= 33 && humidity <= 90, // Kelembaban ada di sini
    isRadiationSesuai: radiation >= 15 && radiation <= 25,
  };

  // 1. SYARAT MUTLAK: HUJAN
  // Jika hujan kurang/banjir, langsung MERAH (Bera), tidak peduli kelembaban bagus.
  if (!criteria.isRainSesuai) {
    return {
      color: "bg-red-300 hover:bg-red-400 border-red-400",
      label: "Masa Bera (Air Tidak Sesuai)",
      count: 0, 
      icon: "✕",
      iconColor: "text-red-700",
      criteria,
      type: "tidakCocok" as const,
    };
  }

  // 2. HITUNG SKOR (Hujan sudah pasti Benar/True di sini)
  const sesuaiCount = Object.values(criteria).filter(Boolean).length;

  // Skor 4/4: Sempurna (Hijau Tua)
  if (sesuaiCount === 4) {
    return {
      color: "bg-green-300 hover:bg-green-400 border-green-400",
      label: "Sangat Cocok (Optimal)",
      count: 4,
      icon: "●",
      iconColor: "text-green-700",
      criteria,
      type: "sangatCocok" as const,
    };
  }
  
  // Skor 3/4: Masih Tanam (Hijau Muda/Kuning)
  // Contoh: Hujan OK, Suhu OK, Solar OK, tapi KELEMBABAN BURUK.
  // Ini masuk kategori "Cukup Cocok".
  if (sesuaiCount === 3) {
    return {
      color: "bg-green-100 hover:bg-green-200 border-green-300",
      label: "Cukup Cocok (Kondisi Baik)",
      count: 3,
      icon: "●",
      iconColor: "text-green-600",
      criteria,
      type: "cukupCocok" as const,
    };
  }

  // Skor < 3: Parameter Pendukung Buruk (Merah)
  // Hujan OK, tapi Kelembaban DAN Suhu jelek (Skor 2).
  // Maka jadi TIDAK COCOK.
  return {
    color: "bg-red-300 hover:bg-red-400 border-red-400",
    label: "Tidak Cocok (Parameter Lingkungan)",
    count: sesuaiCount,
    icon: "✕",
    iconColor: "text-red-700",
    criteria,
    type: "tidakCocok" as const,
  };
};

// ============================================================
// LOGIKA SUMMARY BULANAN
// ============================================================

interface MonthCalendarData {
  month: string;
  year: number;
  monthKey: string;
  firstDayOffset: number;
  daysInMonth: number;
  days: Map<number, "sangatCocok" | "cukupCocok" | "tidakCocok">;
  totalSangatCocok: number;
  totalCukupCocok: number;
  totalTidakCocok: number;
}

interface SuitabilitySummary {
  months: MonthCalendarData[];
  totalSangatCocok: number;
  totalCukupCocok: number;
  totalTidakCocok: number;
}

const getSuitabilitySummary = (data: any[]): SuitabilitySummary => {
  if (!data || data.length === 0) {
    return { months: [], totalSangatCocok: 0, totalCukupCocok: 0, totalTidakCocok: 0 };
  }

  const sortedData = [...data]
    .filter((item) => !item.isPlaceholder)
    .sort((a, b) => new Date(a.forecast_date).getTime() - new Date(b.forecast_date).getTime());

  const monthsMap = new Map<string, MonthCalendarData>();

  sortedData.forEach((item) => {
    const date = new Date(item.forecast_date);
    const monthKey = format(date, "yyyy-MM");
    const monthName = format(date, "MMMM", { locale: id });
    const year = date.getFullYear();
    const day = date.getDate();

    const rain = getParameterValue(item.parameters, ['RR_imputed', 'RR', 'PRECTOTCORR']);
    const temp = getParameterValue(item.parameters, ['TAVG', 'T2M', 'T2M_MAX', 'TMAX']);
    const humidity = getParameterValue(item.parameters, ['RH_AVG_preprocessed', 'RH_AVG', 'RH2M']);
    const radiation = getParameterValue(item.parameters, ['ALLSKY_SFC_SW_DWN']);

    const suitability = getSuitability(rain, temp, humidity, radiation);

    if (!monthsMap.has(monthKey)) {
      const firstDay = startOfMonth(date);
      const firstDayOfWeek = getDay(firstDay);
      const offset = firstDayOfWeek === 0 ? 6 : firstDayOfWeek - 1;
      
      monthsMap.set(monthKey, {
        month: monthName,
        year: year,
        monthKey: monthKey,
        firstDayOffset: offset,
        daysInMonth: getDaysInMonth(date),
        days: new Map(),
        totalSangatCocok: 0,
        totalCukupCocok: 0,
        totalTidakCocok: 0
      });
    }

    const monthData = monthsMap.get(monthKey)!;
    if (suitability.type) {
      monthData.days.set(day, suitability.type);
      if (suitability.type === "sangatCocok") monthData.totalSangatCocok++;
      else if (suitability.type === "cukupCocok") monthData.totalCukupCocok++;
      else monthData.totalTidakCocok++;
    }
  });

  const months = Array.from(monthsMap.values());

  return {
    months,
    totalSangatCocok: months.reduce((sum, m) => sum + m.totalSangatCocok, 0),
    totalCukupCocok: months.reduce((sum, m) => sum + m.totalCukupCocok, 0),
    totalTidakCocok: months.reduce((sum, m) => sum + m.totalTidakCocok, 0),
  };
};

const getPeriodCalendarGrid = (data: any[], startDate: Date, endDate: Date) => {
  const days = eachDayOfInterval({ start: startDate, end: endDate });
  const startOffset = getDay(startDate) === 0 ? 6 : getDay(startDate) - 1;

  const grid: (any | null)[] = Array(startOffset).fill(null);
  days.forEach((date) => {
    const forecast = data.find((item) => format(new Date(item.forecast_date), "yyyy-MM-dd") === format(date, "yyyy-MM-dd"));
    if (forecast) {
      grid.push(forecast);
    } else {
      grid.push({
        forecast_date: date.toISOString(),
        isPlaceholder: true,
        parameters: {},
      });
    }
  });

  const rows: (any | null)[][] = [];
  for (let i = 0; i < grid.length; i += 1) {
    const rowIndex = i % 7; 
    if (!rows[rowIndex]) rows[rowIndex] = [];
    rows[rowIndex].push(grid[i]);
  }

  return rows;
};

const getPeriodData = (data: any[], startDate: Date, endDate: Date) => {
  return data.filter((item) => {
    const itemDate = new Date(item.forecast_date);
    return itemDate >= startDate && itemDate <= endDate;
  });
};

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function PeriodCalendar() {
  const [selectedModel, setSelectedModel] = useState<ModelType>("holt-winters");

  useEffect(() => {
    const event = new CustomEvent('modelChanged', {
      detail: { model: selectedModel }
    });
    window.dispatchEvent(event);
  }, [selectedModel]);

  const {
    data: forecastData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["forecast-data", selectedModel],
    queryFn: async () => {
      const fetchFunc = selectedModel === "lstm" ? getLSTMDaily : getHoltWinterDaily;
      const info = await fetchFunc(1, 10);
      const total = info.total || 365;
      const all = await fetchFunc(1, total);
      
      const sorted = all.items
        .map((item: any) => ({
          ...item,
          forecast_date: new Date(item.forecast_date).toISOString(),
        }))
        .sort((a: any, b: any) => new Date(a.forecast_date).getTime() - new Date(b.forecast_date).getTime());
      
      return sorted.slice(0, 365);
    },
  });

  const { periodRows, periodRanges } = useMemo(() => {
    const emptyState = {
      periodRows: { "Periode-1": [], "Periode-2": [], "Periode-3": [] },
      periodRanges: { "Periode-1": "Memuat...", "Periode-2": "Memuat...", "Periode-3": "Memuat..." },
    };
    
    if (!forecastData || forecastData.length === 0) return emptyState;

    const globalStartDate = new Date(forecastData[0].forecast_date);
    // Logic pembagian periode (4 bulan tanam)
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

  // --- RENDER PERIODE GRID ---
  const renderPeriodGrid = (period: keyof typeof periodRows) => (
    <TabsContent value={period} className="mt-4 space-y-4">
      <Alert>
        <div className="flex items-start gap-2">
          <Info className="h-4 w-4 mt-0.5" />
          <AlertDescription className="flex-1">
            Periode Tanam: <strong>{periodRanges[period]}</strong>
          </AlertDescription>
        </div>
      </Alert>

      <ScrollArea className="w-full rounded-md border">
        <div className="min-w-[900px]">
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50">
                <TableHead className="text-center font-semibold w-24">Hari</TableHead>
                {periodRows[period][0]?.map((_, colIdx) => (
                  <TableHead key={colIdx} className="text-center font-semibold min-w-[80px]">
                    {`Minggu ${colIdx + 1}`}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              <TooltipProvider delayDuration={200}>
                {["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"].map((day, rowIdx) => (
                  <TableRow key={day} className="hover:bg-muted/30">
                    <TableCell className="text-center font-medium text-muted-foreground">
                      {day}
                    </TableCell>

                    {periodRows[period][rowIdx]?.map((dayData, colIdx) => {
                      if (dayData && !dayData.isPlaceholder) {
                        const rain = getParameterValue(dayData.parameters, ['RR_imputed', 'RR', 'PRECTOTCORR']);
                        const temp = getParameterValue(dayData.parameters, ['TAVG', 'T2M', 'T2M_MAX', 'TMAX']);
                        const humidity = getParameterValue(dayData.parameters, ['RH_AVG_preprocessed', 'RH_AVG', 'RH2M']);
                        const radiation = getParameterValue(dayData.parameters, ['ALLSKY_SFC_SW_DWN']);

                        const suitability = getSuitability(rain, temp, humidity, radiation);

                        // --- LOGIKA TOOLTIP ERROR (DIMUNCULKAN KELEMBABANNYA) ---
                        const unsuitable: string[] = [];
                        if (!suitability.criteria?.isRainSesuai) unsuitable.push("Hujan");
                        if (suitability.criteria && !suitability.criteria.isTempSesuai) unsuitable.push("Suhu");
                        // DI SINI KELEMBABAN DIPANGGIL JIKA TIDAK SESUAI
                        if (suitability.criteria && !suitability.criteria.isHumiditySesuai) unsuitable.push("Kelembaban");
                        if (suitability.criteria && !suitability.criteria.isRadiationSesuai) unsuitable.push("Radiasi");

                        return (
                          <Tooltip key={colIdx}>
                            <TooltipTrigger asChild>
                              <TableCell
                                className={clsx(
                                  "text-center h-16 cursor-pointer transition-all duration-200 border relative",
                                  suitability.color
                                )}
                              >
                                <div className="flex flex-col items-center justify-center space-y-1">
                                  <div className="text-xs font-semibold">{format(new Date(dayData.forecast_date), "dd")}</div>
                                  <div className={clsx("text-xl font-bold", suitability.iconColor)}>{suitability.icon}</div>
                                </div>
                                <div className="text-[10px] absolute bottom-1 left-1.5 text-muted-foreground">
                                  {format(new Date(dayData.forecast_date), "MMM", { locale: id })}
                                </div>
                              </TableCell>
                            </TooltipTrigger>
                            <TooltipContent side="top" className="p-3">
                              <div className="space-y-2 min-w-[220px]">
                                <div className="font-semibold pb-2 border-b">
                                  {format(new Date(dayData.forecast_date), "eeee, d MMMM yyyy", { locale: id })}
                                </div>
                                <div className="space-y-1.5 text-sm">
                                  <div className={clsx("flex items-center justify-between gap-2", suitability.criteria && !suitability.criteria.isRainSesuai && "text-red-600")}>
                                    <div className="flex items-center gap-2">
                                      <Droplets className="w-4 h-4 text-blue-500" />
                                      <span>Hujan: <strong>{rain.toFixed(2)} mm</strong></span>
                                    </div>
                                    {suitability.criteria && !suitability.criteria.isRainSesuai && <XCircle className="w-3 h-3 text-red-500" />}
                                  </div>
                                  
                                  <div className={clsx("flex items-center justify-between gap-2", suitability.criteria && !suitability.criteria.isTempSesuai && "text-red-600")}>
                                    <div className="flex items-center gap-2">
                                      <Thermometer className="w-4 h-4 text-orange-500" />
                                      <span>Suhu: <strong>{temp.toFixed(2)}°C</strong></span>
                                    </div>
                                    {suitability.criteria && !suitability.criteria.isTempSesuai && <XCircle className="w-3 h-3 text-red-500" />}
                                  </div>

                                  {/* TAMPILAN INDIKATOR KELEMBABAN */}
                                  <div className={clsx("flex items-center justify-between gap-2", suitability.criteria && !suitability.criteria.isHumiditySesuai && "text-red-600")}>
                                    <div className="flex items-center gap-2">
                                      <Wind className="w-4 h-4 text-cyan-500" />
                                      <span>Kelembaban: <strong>{humidity.toFixed(2)}%</strong></span>
                                    </div>
                                    {suitability.criteria && !suitability.criteria.isHumiditySesuai && <XCircle className="w-3 h-3 text-red-500" />}
                                  </div>

                                  <div className={clsx("flex items-center justify-between gap-2", suitability.criteria && !suitability.criteria.isRadiationSesuai && "text-red-600")}>
                                    <div className="flex items-center gap-2">
                                      <Sun className="w-4 h-4 text-yellow-500" />
                                      <span>Radiasi: <strong>{radiation.toFixed(2)} MJ/m²</strong></span>
                                    </div>
                                    {suitability.criteria && !suitability.criteria.isRadiationSesuai && <XCircle className="w-3 h-3 text-red-500" />}
                                  </div>
                                </div>
                                
                                <div className="pt-2 border-t">
                                  <p className="font-bold text-center">
                                    {suitability.label}
                                  </p>
                                  {unsuitable.length > 0 && (
                                    <p className="text-xs text-red-600 text-center mt-1">
                                      Tidak sesuai: {unsuitable.join(", ")}
                                    </p>
                                  )}
                                </div>
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        );
                      } else {
                        return (
                          <TableCell key={colIdx} className="text-center h-16 bg-muted/30">
                            <span className="text-muted-foreground">-</span>
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
        <Card>
          <CardHeader>
            <Skeleton className="h-8 w-64" />
            <Skeleton className="h-4 w-96 mt-2" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center py-12 gap-2">
              <Loader2 className="w-5 h-5 animate-spin text-primary" />
              <span className="text-muted-foreground">Memuat data prediksi...</span>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="border-destructive">
        <CardContent className="pt-6">
          <Alert variant="destructive">
            <AlertDescription>
              Terjadi kesalahan saat memuat data prediksi. Silakan coba lagi.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <CardTitle className="flex items-center gap-2">
                <Calendar className="w-5 h-5" />
                Kalender Prediksi Cuaca per Periode
              </CardTitle>
              <CardDescription>
                Prediksi cuaca berdasarkan model {selectedModel === "lstm" ? "LSTM" : "Holt Winters"}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs">
                Model Prediksi
              </Badge>
              <Select value={selectedModel} onValueChange={(value: ModelType) => setSelectedModel(value)}>
                <SelectTrigger className="w-44">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="holt-winters">Holt Winters</SelectItem>
                  <SelectItem value="lstm">LSTM</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="Periode-1" className="w-full">
            <TabsList className="grid w-full grid-cols-3 h-auto">
              {["Periode-1", "Periode-2", "Periode-3"].map((period) => (
                <TabsTrigger key={period} value={period} className="flex flex-col py-2">
                  <span className="font-semibold">{period}</span>
                  <span className="text-xs text-muted-foreground">{periodRanges[period as keyof typeof periodRanges]}</span>
                </TabsTrigger>
              ))}
            </TabsList>
            {["Periode-1", "Periode-2", "Periode-3"].map((period) => (
              <React.Fragment key={period}>
                {renderPeriodGrid(period as keyof typeof periodRows)}
              </React.Fragment>
            ))}
          </Tabs>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Keterangan Status</CardTitle>
          <CardDescription>
            Indikator kondisi cuaca untuk penanaman padi
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="flex items-center gap-2 p-2.5 rounded-lg border bg-green-300/50">
              <div className="w-7 h-7 bg-green-400 border border-green-500 rounded-md flex items-center justify-center text-green-900 font-bold text-lg">●</div>
              <div className="space-y-1">
                <p className="font-medium text-sm leading-none">Sangat Cocok</p>
                <p className="text-xs text-muted-foreground leading-none">4/4 parameter sesuai</p>
              </div>
            </div>
            <div className="flex items-center gap-2 p-2.5 rounded-lg border bg-green-100/50">
              <div className="w-7 h-7 bg-green-200 border border-green-300 rounded-md flex items-center justify-center text-green-800 font-bold text-lg">●</div>
              <div className="space-y-1">
                <p className="font-medium text-sm leading-none">Cukup Cocok</p>
                <p className="text-xs text-muted-foreground leading-none">3/4 (Hujan Wajib + 2 Lainnya)</p>
              </div>
            </div>
            <div className="flex items-center gap-2 p-2.5 rounded-lg border bg-red-300/50">
              <div className="w-7 h-7 bg-red-400 border border-red-500 rounded-md flex items-center justify-center text-red-900 font-bold text-lg">✕</div>
              <div className="space-y-1">
                <p className="font-medium text-sm leading-none">Tidak Cocok (Bera)</p>
                <p className="text-xs text-muted-foreground leading-none">Hujan tidak sesuai / &lt;3 parameter</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}