"use client";

import React, { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getHoltWinterDaily, getLSTMDaily } from "@/lib/fetch/files.fetch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useMemo, useState } from "react";
import clsx from "clsx";
import { format, eachDayOfInterval, getDay, addMonths, subDays, addDays, startOfMonth, getDaysInMonth } from "date-fns";
import { id } from "date-fns/locale";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar, Droplets, Thermometer, Wind, Sun, Info, Loader2, XCircle, ChevronDown, CalendarDays } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";

type ModelType = "holt-winters" | "lstm";

// ============================================================
// HELPER FUNCTIONS
// ============================================================

// Helper function untuk mengambil nilai parameter dengan fallback
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

// ============================================================
// HELPER FUNCTIONS
// ============================================================

const getSuitability = (rain: number, temp: number, humidity: number, radiation: number) => {
  const criteria = {
    isRainSesuai: rain >= 5.7 && rain <= 16.7,
    isTempSesuai: temp >= 24 && temp <= 29,
    isHumiditySesuai: humidity >= 33 && humidity <= 90,  // Sesuai kriteria penelitian: 33-90%
    isRadiationSesuai: radiation >= 15 && radiation <= 25,  // Sesuai kriteria penelitian: 15-25 MJ/m²/hari
  };

  const sesuaiCount = Object.values(criteria).filter(Boolean).length;

  if (sesuaiCount === 4) {
    return {
      color: "bg-green-300 hover:bg-green-400 border-green-400",
      label: "Sangat Cocok",
      count: 4,
      icon: "●",
      iconColor: "text-green-700",
      criteria,
      type: "sangatCocok" as const,
    };
  }
  if (sesuaiCount === 3) {
    return {
      color: "bg-green-100 hover:bg-green-200 border-green-300",
      label: "Cukup Cocok",
      count: 3,
      icon: "●",
      iconColor: "text-green-600",
      criteria,
      type: "cukupCocok" as const,
    };
  }
  return {
    color: "bg-red-300 hover:bg-red-400 border-red-400",
    label: "Tidak Cocok",
    count: sesuaiCount,
    icon: "✕",
    iconColor: "text-red-700",
    criteria,
    type: "tidakCocok" as const,
  };
};

// ============================================================
// Interface untuk Summary per Bulan
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

// ============================================================
// Function untuk mendapatkan ringkasan per bulan dengan data kalender
// ============================================================

const getSuitabilitySummary = (data: any[]): SuitabilitySummary => {
  if (!data || data.length === 0) {
    return {
      months: [],
      totalSangatCocok: 0,
      totalCukupCocok: 0,
      totalTidakCocok: 0,
    };
  }

  const sortedData = [...data]
    .filter((item) => !item.isPlaceholder)
    .sort((a, b) => new Date(a.forecast_date).getTime() - new Date(b.forecast_date).getTime());

  const monthsMap = new Map<string, {
    month: string;
    year: number;
    firstDayOffset: number;
    daysInMonth: number;
    days: Map<number, "sangatCocok" | "cukupCocok" | "tidakCocok">;
  }>();

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
        firstDayOffset: offset,
        daysInMonth: getDaysInMonth(date),
        days: new Map(),
      });
    }

    const monthData = monthsMap.get(monthKey)!;
    if (suitability.type) {
      monthData.days.set(day, suitability.type);
    }
  });

  const months: MonthCalendarData[] = Array.from(monthsMap.entries()).map(([key, data]) => {
    let totalSangatCocok = 0;
    let totalCukupCocok = 0;
    let totalTidakCocok = 0;
    
    data.days.forEach((type) => {
      if (type === "sangatCocok") totalSangatCocok++;
      else if (type === "cukupCocok") totalCukupCocok++;
      else totalTidakCocok++;
    });

    return {
      month: data.month,
      year: data.year,
      monthKey: key,
      firstDayOffset: data.firstDayOffset,
      daysInMonth: data.daysInMonth,
      days: data.days,
      totalSangatCocok,
      totalCukupCocok,
      totalTidakCocok,
    };
  });

  return {
    months,
    totalSangatCocok: months.reduce((sum, m) => sum + m.totalSangatCocok, 0),
    totalCukupCocok: months.reduce((sum, m) => sum + m.totalCukupCocok, 0),
    totalTidakCocok: months.reduce((sum, m) => sum + m.totalTidakCocok, 0),
  };
};

// ============================================================
// Mini Calendar Component per Bulan
// ============================================================

interface MiniCalendarProps {
  monthData: MonthCalendarData;
}

const MiniCalendar: React.FC<MiniCalendarProps> = ({ monthData }) => {
  const dayNames = ["S", "S", "R", "K", "J", "S", "M"];
  
  const calendarGrid: (number | null)[] = [];
  
  for (let i = 0; i < monthData.firstDayOffset; i++) {
    calendarGrid.push(null);
  }
  
  for (let day = 1; day <= monthData.daysInMonth; day++) {
    calendarGrid.push(day);
  }
  
  const weeks: (number | null)[][] = [];
  for (let i = 0; i < calendarGrid.length; i += 7) {
    weeks.push(calendarGrid.slice(i, i + 7));
  }
  
  if (weeks.length > 0 && weeks[weeks.length - 1].length < 7) {
    const lastWeek = weeks[weeks.length - 1];
    while (lastWeek.length < 7) {
      lastWeek.push(null);
    }
  }

  const getDayColor = (day: number | null) => {
    if (day === null) return "bg-transparent";
    const type = monthData.days.get(day);
    if (!type) return "bg-gray-100 text-gray-400";
    
    switch (type) {
      case "sangatCocok":
        return "bg-green-400 text-green-900 font-medium";
      case "cukupCocok":
        return "bg-green-200 text-green-800 font-medium";
      case "tidakCocok":
        return "bg-red-400 text-red-900 font-medium";
      default:
        return "bg-gray-100";
    }
  };

  // Tentukan apakah bulan ini cocok untuk tanam atau tidak
  const isBulanCocokTanam = monthData.totalTidakCocok <= 15;
  const headerBgColor = isBulanCocokTanam 
    ? "bg-green-100 border-green-300" 
    : "bg-red-100 border-red-300";

  return (
    <div className="border rounded-lg overflow-hidden">
      {/* Month Header */}
      <div className={clsx("px-2 py-1.5 border-b", headerBgColor)}>
        <div className="flex items-center justify-between">
          <span className="font-semibold text-xs">{monthData.month} {monthData.year}</span>
          <div className="flex items-center gap-1.5 text-[10px]">
            <span className="flex items-center gap-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400"></span>
              {monthData.totalSangatCocok}
            </span>
            <span className="flex items-center gap-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-green-200"></span>
              {monthData.totalCukupCocok}
            </span>
            <span className="flex items-center gap-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-red-400"></span>
              {monthData.totalTidakCocok}
            </span>
          </div>
        </div>
      </div>
      
      {/* Day Names */}
      <div className="grid grid-cols-7 bg-muted/30">
        {dayNames.map((name, idx) => (
          <div key={idx} className="text-center text-[10px] font-medium text-muted-foreground py-1">
            {name}
          </div>
        ))}
      </div>
      
      {/* Calendar Grid */}
      <div className="p-0.5">
        {weeks.map((week, weekIdx) => (
          <div key={weekIdx} className="grid grid-cols-7 gap-0.5">
            {week.map((day, dayIdx) => (
              <div
                key={dayIdx}
                className={clsx(
                  "aspect-square flex items-center justify-center text-[10px] rounded-sm",
                  getDayColor(day)
                )}
              >
                {day}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

// // ============================================================
// // Yearly Calendar Component - 12 Bulan dalam 1 Tampilan
// // ============================================================

// interface YearlyCalendarProps {
//   summary: SuitabilitySummary;
//   title?: string;
// }

// const YearlyCalendar: React.FC<YearlyCalendarProps> = ({ summary, title }) => {
//   const [isOpen, setIsOpen] = useState(false); // Ubah default ke false (tertutup)

//   // Hitung rentang tahun dari data
//   const yearRange = useMemo(() => {
//     if (summary.months.length === 0) return "";
//     const firstMonth = summary.months[0];
//     const lastMonth = summary.months[summary.months.length - 1];
//     if (firstMonth.year === lastMonth.year) {
//       return `${firstMonth.year}`;
//     }
//     return `${firstMonth.year} - ${lastMonth.year}`;
//   }, [summary.months]);

//   return (
//     <Card>
//       <Collapsible open={isOpen} onOpenChange={setIsOpen}>
//         <CardHeader className="pb-3">
//           <div className="flex items-center justify-between">
//             <div className="flex items-center gap-2">
//               <CalendarDays className="w-5 h-5" />
//               <div>
//                 <CardTitle className="text-base">
//                   {title || `Kalender Tahunan ${yearRange}`}
//                 </CardTitle>
//                 <CardDescription className="text-xs">
//                   Ringkasan kesesuaian tanam selama {summary.months.length} bulan
//                 </CardDescription>
//               </div>
//             </div>
            
//             <div className="flex items-center gap-3">
//               {/* Summary Stats */}
//               <div className="hidden sm:flex items-center gap-3">
//                 <div className="flex items-center gap-1.5">
//                   <div className="w-3 h-3 rounded-full bg-green-400" />
//                   <span className="text-sm font-medium">{summary.totalSangatCocok}</span>
//                 </div>
//                 <div className="flex items-center gap-1.5">
//                   <div className="w-3 h-3 rounded-full bg-green-200" />
//                   <span className="text-sm font-medium">{summary.totalCukupCocok}</span>
//                 </div>
//                 <div className="flex items-center gap-1.5">
//                   <div className="w-3 h-3 rounded-full bg-red-400" />
//                   <span className="text-sm font-medium">{summary.totalTidakCocok}</span>
//                 </div>
//               </div>
              
//               <CollapsibleTrigger asChild>
//                 <Button variant="ghost" size="sm" className="gap-1">
//                   <span className="text-xs">{isOpen ? "Tutup Detail" : "Lihat Detail"}</span>
//                   <ChevronDown className={clsx("h-4 w-4 transition-transform", isOpen && "rotate-180")} />
//                 </Button>
//               </CollapsibleTrigger>
//             </div>
//           </div>
          
//           {/* Mobile Stats */}
//           <div className="flex sm:hidden items-center gap-4 mt-2">
//             <div className="flex items-center gap-1.5">
//               <div className="w-3 h-3 rounded-full bg-green-400" />
//               <span className="text-sm">Sangat Cocok: <strong>{summary.totalSangatCocok}</strong></span>
//             </div>
//             <div className="flex items-center gap-1.5">
//               <div className="w-3 h-3 rounded-full bg-green-200" />
//               <span className="text-sm">Cukup: <strong>{summary.totalCukupCocok}</strong></span>
//             </div>
//             <div className="flex items-center gap-1.5">
//               <div className="w-3 h-3 rounded-full bg-red-400" />
//               <span className="text-sm">Tidak: <strong>{summary.totalTidakCocok}</strong></span>
//             </div>
//           </div>
//         </CardHeader>

//         {/* Ringkasan Per Bulan (Tampil saat ditutup) */}
//         {!isOpen && (
//           <CardContent className="pt-0">
//             <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
//               {summary.months.map((monthData) => {
//                 const isBulanCocokTanam = monthData.totalTidakCocok <= 15;
                
//                 return (
//                   <div 
//                     key={monthData.monthKey}
//                     className={clsx(
//                       "border-2 rounded-lg p-4 transition-all hover:shadow-md",
//                       isBulanCocokTanam 
//                         ? "border-green-400 bg-green-50/50" 
//                         : "border-red-400 bg-red-50/50"
//                     )}
//                   >
//                     <div className="space-y-3">
//                       {/* Nama Bulan */}
//                       <div className="font-bold text-base text-center">
//                         {monthData.month}
//                       </div>
                      
//                       {/* Status Badge */}
//                       <div className="flex justify-center">
//                         <Badge 
//                           variant={isBulanCocokTanam ? "default" : "destructive"}
//                           className="px-3 py-1"
//                         >
//                           {isBulanCocokTanam ? "✓ Cocok Tanam" : "✕ Tidak Cocok"}
//                         </Badge>
//                       </div>
                      
//                       {/* Statistik Sederhana */}
//                       <div className="text-center text-sm text-muted-foreground">
//                         <div className="font-medium">
//                           {monthData.totalSangatCocok + monthData.totalCukupCocok} dari {monthData.daysInMonth} hari
//                         </div>
//                         <div className="text-xs mt-0.5">
//                           kondisi baik
//                         </div>
//                       </div>
//                     </div>
//                   </div>
//                 );
//               })}
//             </div>
            
//             {/* Info Helper */}
//             <div className="mt-4 pt-3 border-t">
//               <p className="text-xs text-center text-muted-foreground">
//                 Klik <strong>Lihat Detail</strong> untuk melihat kalender lengkap per tanggal
//               </p>
//             </div>
//           </CardContent>
//         )}

//         {/* Detail Kalender (Tampil saat dibuka) */}
//         <CollapsibleContent>
//           <CardContent className="pt-0">
//             {/* Grid 12 Bulan: 4 kolom x 3 baris */}
//             <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-2">
//               {summary.months.map((monthData) => (
//                 <MiniCalendar key={monthData.monthKey} monthData={monthData} />
//               ))}
//             </div>
            
//             {/* Legend */}
//             <div className="flex flex-wrap items-center justify-center gap-4 mt-4 pt-3 border-t text-xs text-muted-foreground">
//               <div className="flex items-center gap-1.5">
//                 <div className="w-4 h-4 rounded-sm bg-green-400"></div>
//                 <span>Sangat Cocok (4/4)</span>
//               </div>
//               <div className="flex items-center gap-1.5">
//                 <div className="w-4 h-4 rounded-sm bg-green-200"></div>
//                 <span>Cukup Cocok (3/4)</span>
//               </div>
//               <div className="flex items-center gap-1.5">
//                 <div className="w-4 h-4 rounded-sm bg-red-400"></div>
//                 <span>Tidak Cocok (&lt;3)</span>
//               </div>
//               <div className="flex items-center gap-1.5">
//                 <div className="w-4 h-4 rounded-sm bg-gray-100 border"></div>
//                 <span>Tidak ada data</span>
//               </div>
//             </div>
//           </CardContent>
//         </CollapsibleContent>
//       </Collapsible>
//     </Card>
//   );
// };

// ============================================================
// CALENDAR GRID FUNCTIONS
// ============================================================

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
    const rowIndex = i % 7; // 0=Senin, 1=Selasa, ...
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
      if (selectedModel === "lstm") {
        const info = await getLSTMDaily(1, 10);
        const total = info.total || 365;
        const all = await getLSTMDaily(1, total);
        
        const sorted = all.items
          .map((item: any) => ({
            ...item,
            forecast_date: new Date(item.forecast_date).toISOString(),
          }))
          .sort((a: any, b: any) => new Date(a.forecast_date).getTime() - new Date(b.forecast_date).getTime());
        
        const sliced = sorted.slice(0, 365);
        
        console.log("=== LSTM RESULT ===");
        console.log("Total from DB:", total);
        console.log("After slice:", sliced.length);
        console.log("First date:", sliced[0]?.forecast_date);
        console.log("Last date:", sliced[sliced.length - 1]?.forecast_date);
        
        return sliced;
      } else {
        const info = await getHoltWinterDaily(1, 10);
        const total = info.total || 365;
        
        console.log("=== HOLT-WINTERS ===");
        console.log("Total in DB:", total);
        
        const all = await getHoltWinterDaily(1, total);
        
        const sorted = all.items
          .map((item: any) => ({
            ...item,
            forecast_date: new Date(item.forecast_date).toISOString(),
          }))
          .sort((a: any, b: any) => new Date(a.forecast_date).getTime() - new Date(b.forecast_date).getTime());
        
        const sliced = sorted.slice(0, 365);
        
        console.log("After sorting - Total:", sorted.length);
        console.log("After slice (365) - Total:", sliced.length);
        console.log("First date:", sliced[0]?.forecast_date);
        console.log("Last date:", sliced[sliced.length - 1]?.forecast_date);
        
        return sliced;
      }
    },
  });

  const yearlySummary = useMemo(() => {
    if (!forecastData || forecastData.length === 0) return null;
    return getSuitabilitySummary(forecastData);
  }, [forecastData]);

  const { periodRows, periodRanges, periodSummaries } = useMemo(() => {
    const emptyState = {
      periodRows: { "Periode-1": [], "Periode-2": [], "Periode-3": [] },
      periodRanges: { "Periode-1": "Memuat...", "Periode-2": "Memuat...", "Periode-3": "Memuat..." },
      periodSummaries: { "Periode-1": null, "Periode-2": null, "Periode-3": null },
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
    const newPeriodSummaries: Record<string, SuitabilitySummary | null> = {};

    Object.keys(dynamicPeriods).forEach((period) => {
      const { start, end } = dynamicPeriods[period as keyof typeof dynamicPeriods];
      newPeriodRows[period] = getPeriodCalendarGrid(forecastData, start, end);
      newPeriodRanges[period] = `${format(start, "d MMM yyyy", { locale: id })} - ${format(end, "d MMM yyyy", { locale: id })}`;
      
      const periodData = getPeriodData(forecastData, start, end);
      newPeriodSummaries[period] = getSuitabilitySummary(periodData);
    });

    return { 
      periodRows: newPeriodRows, 
      periodRanges: newPeriodRanges, 
      periodSummaries: newPeriodSummaries 
    };
  }, [forecastData]);

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

                        const unsuitable: string[] = [];
                        if (suitability.criteria && !suitability.criteria.isRainSesuai) unsuitable.push("Curah Hujan");
                        if (suitability.criteria && !suitability.criteria.isTempSesuai) unsuitable.push("Suhu");
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
                                    <div className="flex items-center gap-1">
                                      {suitability.criteria && !suitability.criteria.isRainSesuai && <XCircle className="w-3 h-3 text-red-500" />}
                                      <span className="text-xs text-muted-foreground">(5.7-16.7)</span>
                                    </div>
                                  </div>
                                  <div className={clsx("flex items-center justify-between gap-2", suitability.criteria && !suitability.criteria.isTempSesuai && "text-red-600")}>
                                    <div className="flex items-center gap-2">
                                      <Thermometer className="w-4 h-4 text-orange-500" />
                                      <span>Suhu: <strong>{temp.toFixed(2)}°C</strong></span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                      {suitability.criteria && !suitability.criteria.isTempSesuai && <XCircle className="w-3 h-3 text-red-500" />}
                                      <span className="text-xs text-muted-foreground">(24-29)</span>
                                    </div>
                                  </div>
                                  <div className={clsx("flex items-center justify-between gap-2", suitability.criteria && !suitability.criteria.isHumiditySesuai && "text-red-600")}>
                                    <div className="flex items-center gap-2">
                                      <Wind className="w-4 h-4 text-cyan-500" />
                                      <span>Kelembaban: <strong>{humidity.toFixed(2)}%</strong></span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                      {suitability.criteria && !suitability.criteria.isHumiditySesuai && <XCircle className="w-3 h-3 text-red-500" />}
                                      <span className="text-xs text-muted-foreground">(33-90)</span>
                                    </div>
                                  </div>
                                  <div className={clsx("flex items-center justify-between gap-2", suitability.criteria && !suitability.criteria.isRadiationSesuai && "text-red-600")}>
                                    <div className="flex items-center gap-2">
                                      <Sun className="w-4 h-4 text-yellow-500" />
                                      <span>Radiasi: <strong>{radiation.toFixed(2)} MJ/m²</strong></span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                      {suitability.criteria && !suitability.criteria.isRadiationSesuai && <XCircle className="w-3 h-3 text-red-500" />}
                                      <span className="text-xs text-muted-foreground">(15-25)</span>
                                    </div>
                                  </div>
                                </div>
                                <div className="pt-2 border-t">
                                  <p className="font-bold text-center">
                                    {suitability.label} ({suitability.count}/4)
                                  </p>
                                  {unsuitable.length > 0 && (
                                    <p className="text-xs text-red-600 text-center mt-1">
                                      Tidak sesuai: {unsuitable.join(", ")}
                                    </p>
                                  )}
                                </div>
                              </div>
                              {/* --- Akhir Tooltip Diperbarui --- */}
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
      {/* Kalender Tahunan - 12 Bulan
      {yearlySummary && (
        <YearlyCalendar 
          summary={yearlySummary} 
          title={`Ringkasan Kalender Tanam 1 Tahun`}
        />
      )} */}

      {/* Kalender Per Periode */}
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

      {/* Keterangan Status */}
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
              <div className="w-7 h-7 bg-green-400 border border-green-500 rounded-md flex items-center justify-center text-green-900 font-bold text-lg">
                ●
              </div>
              <div className="space-y-1">
                <p className="font-medium text-sm leading-none">Sangat Cocok</p>
                <p className="text-xs text-muted-foreground leading-none">4/4 parameter sesuai</p>
              </div>
            </div>
            {/* Ganti dari yellow ke hijau muda */}
            <div className="flex items-center gap-2 p-2.5 rounded-lg border bg-green-100/50">
              <div className="w-7 h-7 bg-green-200 border border-green-300 rounded-md flex items-center justify-center text-green-800 font-bold text-lg">
                ●
              </div>
              <div className="space-y-1">
                <p className="font-medium text-sm leading-none">Cukup Cocok</p>
                <p className="text-xs text-muted-foreground leading-none">3/4 parameter sesuai</p>
              </div>
            </div>
            <div className="flex items-center gap-2 p-2.5 rounded-lg border bg-red-300/50">
              <div className="w-7 h-7 bg-red-400 border border-red-500 rounded-md flex items-center justify-center text-red-900 font-bold text-lg">
                ✕
              </div>
              <div className="space-y-1">
                <p className="font-medium text-sm leading-none">Tidak Cocok</p>
                <p className="text-xs text-muted-foreground leading-none">&lt;3 parameter sesuai</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
