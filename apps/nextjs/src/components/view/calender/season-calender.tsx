"use client";

import { useQuery } from "@tanstack/react-query";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { getHoltWinterSummary } from "@/lib/fetch/files.fetch";

interface SummaryItem {
  _id: string;
  month: string; 
  kt_period: string; // "KT-1" | "KT-2" | "KT-3"
  status: string;
  parameters: {
    RR_imputed?: {
      avg: number;
    };
  };
  reason: string;
}

function SeasonalCalendarTabs() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["hw-summary"],
    queryFn: getHoltWinterSummary,
  });

  if (isLoading) {
    return (
      <div className="space-y-4 p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-64 bg-gray-200 rounded" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4 p-6">
        <div className="text-red-500 p-4 border border-red-200 rounded">Gagal memuat data: {error instanceof Error ? error.message : "Unknown error"}</div>
      </div>
    );
  }

  const rawData = data?.items as SummaryItem[];

  const groupedByKT = rawData.reduce<Record<string, SummaryItem[]>>((acc, item) => {
    if (!acc[item.kt_period]) acc[item.kt_period] = [];
    acc[item.kt_period].push(item);
    return acc;
  }, {});

  const getStatusColor = (status: string) => {
    const clean = status.toLowerCase();
    if (clean === "tanam") return "bg-emerald-500 text-white";
    if (clean === "panen") return "bg-yellow-100 text-yellow-800";
    if (clean === "rehat" || clean === "istirahat") return "bg-gray-100 text-gray-600";
    if (clean === "tidak cocok tanam") return "bg-red-100 text-red-800";
    return "bg-gray-100 text-gray-400"; // fallback
  };

  return (
    <div className="w-full space-y-4 p-6">
      <Tabs defaultValue="KT-1" className="w-full">
        <TabsList className="mb-4">
          {["KT-1", "KT-2", "KT-3"].map((kt) => (
            <TabsTrigger key={kt} value={kt}>
              {kt}
            </TabsTrigger>
          ))}
        </TabsList>

        {["KT-1", "KT-2", "KT-3"].map((kt) => {
          const months = groupedByKT[kt]?.sort((a, b) => a.month.localeCompare(b.month)) || [];

          return (
            <TabsContent key={kt} value={kt}>
              <Table className="min-w-full bg-background">
                <TableHeader>
                  <TableRow className="border-y-0 hover:bg-transparent [&>:not(:last-child)]:border-r">
                    {months.map((m) => (
                      <TableHead key={m.month} className="text-center text-xs py-2">
                        {m.month}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow className="*:border-border [&>:not(:last-child)]:border-r">
                    {months.map((m, i) => (
                      <TableCell key={`${m.month}-${i}`} className={`text-center p-2 text-xs ${getStatusColor(m.status)}`} title={`Status: ${m.status}\nCurah Hujan: ${m.parameters?.RR_imputed?.avg?.toFixed(1) || "-"} mm`}>
                        <div className="font-medium">{m.status}</div>
                        {m.parameters?.RR_imputed && <div className="opacity-75">{m.parameters.RR_imputed.avg.toFixed(0)}mm</div>}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableBody>
              </Table>
            </TabsContent>
          );
        })}
      </Tabs>
    </div>
  );
}

export default SeasonalCalendarTabs;
