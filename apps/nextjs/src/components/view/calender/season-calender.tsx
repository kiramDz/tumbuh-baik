"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

import { useQuery } from "@tanstack/react-query";
import { getBmkgSummary } from "@/lib/fetch/files.fetch";

interface PlantSummaryData {
  _id: string;
  month: string; // format: YYYY-MM
  curah_hujan_total: number;
  kelembapan_avg: number;
  status: string;
  timestamp: string;
}

interface MonthSummaryData {
  month: string;
  monthNumber: string;
  status: string;
  curah_hujan: number;
  kelembapan: number;
  hasData: boolean;
}

function SeasonalCalendarTabs() {
  const {
    data: summaryData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["bmkg-summary"],
    queryFn: getBmkgSummary,
  });

  const currentYear = new Date().getFullYear();

  const processMonthlyData = (data: PlantSummaryData[]): MonthSummaryData[] => {
    const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const dataMap = new Map<string, PlantSummaryData>();

    data?.forEach((item) => {
      const [year, month] = item.month.split("-");
      if (Number(year) === currentYear) {
        dataMap.set(month, item);
      }
    });

    return monthNames.map((monthName, index) => {
      const monthNumber = String(index + 1).padStart(2, "0");
      const monthData = dataMap.get(monthNumber);

      return {
        month: monthName,
        monthNumber,
        status: monthData?.status || "no data",
        curah_hujan: monthData?.curah_hujan_total || 0,
        kelembapan: monthData?.kelembapan_avg || 0,
        hasData: !!monthData,
      };
    });
  };

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

  // Error state
  if (error) {
    return (
      <div className="space-y-4">
        <div className="text-red-500 p-4 border border-red-200 rounded">Error loading data: {error instanceof Error ? error.message : "Unknown error"}</div>
      </div>
    );
  }

  const monthlyData = processMonthlyData(summaryData || []);

  const groupByKT = (data: MonthSummaryData[]) => {
    const kt1 = ["09", "10", "11", "12", "01", "02"];
    const kt2 = ["03", "04", "05"];
    const kt3 = ["06", "07", "08"];

    return {
      "KT-1": data.filter((d) => kt1.includes(d.monthNumber)),
      "KT-2": data.filter((d) => kt2.includes(d.monthNumber)),
      "KT-3": data.filter((d) => kt3.includes(d.monthNumber)),
    };
  };

  const seasonalData = groupByKT(monthlyData);

  const getStatusColor = (status: string, hasData: boolean) => {
    if (!hasData) return "bg-gray-100 text-gray-400";
    const statusColorMap: Record<string, string> = {
      "cocok tanam": "bg-emerald-100 text-emerald-800",
      "sangat cocok tanam": "bg-emerald-500 text-white",
      tanam: "bg-emerald-500 text-white",
      "tidak cocok tanam": "bg-red-100 text-red-800",
      panen: "bg-yellow-100 text-yellow-800",
      istirahat: "bg-gray-100 text-gray-600",
    };
    return statusColorMap[status.toLowerCase()] || "bg-gray-100 text-gray-600";
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

        {Object.entries(seasonalData).map(([kt, months]) => (
          <TabsContent key={kt} value={kt}>
            <Table className="min-w-full bg-background">
              <TableHeader>
                <TableRow className="border-y-0 hover:bg-transparent [&>:not(:last-child)]:border-r">
                  {months.map((month) => (
                    <TableHead key={month.month} className="text-center text-xs py-2">
                      {month.month}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow className="*:border-border [&>:not(:last-child)]:border-r">
                  {months.map((month, index) => (
                    <TableCell
                      key={`${month.month}-${index}`}
                      className={`text-center p-2 text-xs ${getStatusColor(month.status, month.hasData)}`}
                      title={month.hasData ? `Curah Hujan: ${month.curah_hujan.toFixed(1)}mm\nKelembapan: ${month.kelembapan.toFixed(1)}%` : "Data tidak tersedia"}
                    >
                      <div className="font-medium">{month.hasData ? month.status : "No Data"}</div>
                      {month.hasData && <div className="opacity-75">{month.curah_hujan.toFixed(0)}mm</div>}
                    </TableCell>
                  ))}
                </TableRow>
              </TableBody>
            </Table>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}

export default SeasonalCalendarTabs;
