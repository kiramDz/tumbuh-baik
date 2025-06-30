import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { useQuery } from "@tanstack/react-query";
import { getBmkgSummary } from "@/lib/fetch/files.fetch";
import CalendarSeedPlanner from "./daily-calender";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";

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
  monthNumber: string; // Format: "01", "02"
  status: string;
  curah_hujan: number;
  kelembapan: number;
  hasData: boolean;
}

function YearlyCalender() {
  const {
    data: summaryData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["bmkg-summary"],
    queryFn: getBmkgSummary,
  });

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

  const currentYear = new Date().getFullYear();

  // Process data untuk membuat struktur bulan
  const processMonthlyData = (data: PlantSummaryData[]) => {
    const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

    const dataMap = new Map<string, PlantSummaryData>();
    data?.forEach((item) => {
      const [year, month] = item.month.split("-");
      if (Number(year) === currentYear) {
        dataMap.set(month, item);
      }
    });

    const processedData = monthNames.map((monthName, index) => {
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

    return processedData;
  };

  const monthlyData = processMonthlyData(summaryData || []);

  const getPlantingSeasons = () => {
    const seasons = [];
    let currentSeason: MonthSummaryData[] = [];
    let seasonNumber = 1;

    monthlyData.forEach((month) => {
      if (month.status === "sangat cocok tanam") {
        currentSeason.push(month);
      } else {
        if (currentSeason.length > 0) {
          seasons.push({
            name: `Musim Tanam ${seasonNumber}`,
            months: [...currentSeason],
          });
          currentSeason = [];
          seasonNumber++;
        }
      }
    });

    if (currentSeason.length > 0) {
      seasons.push({
        name: `Musim Tanam ${seasonNumber}`,
        months: [...currentSeason],
      });
    }

    if (seasons.length === 0) {
      const firstHalf = monthlyData.slice(0, 6);
      const secondHalf = monthlyData.slice(6);

      return [
        { name: "Periode 1", months: firstHalf },
        { name: "Periode 2", months: secondHalf },
      ];
    }

    return seasons;
  };

  const plantingSeasons = getPlantingSeasons();

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
    <div className="spaye-y-4">
      <div className="flex items-center w-full justify-between">
        <div className="flex gap-2">
          <Badge className="rounded-md bg-green-300">Tanam</Badge>
          <Badge className="rounded-md bg-yellow-500">Panen</Badge>
          <Badge variant="destructive" className="rounded-md">
            Tidak Cocok tanam
          </Badge>
        </div>
      </div>
      <ScrollArea className="w-full overflow-auto whitespace-nowrap ">
        <Table className="bg-background mt-6 min-w-[900px]">
          <TableHeader>
            <TableRow className="border-y-0 *:border-border hover:bg-transparent [&>:not(:last-child)]:border-r">
              <TableCell></TableCell>
              {plantingSeasons.map((season, index) => (
                <TableHead key={index} className="border-b border-border text-center" colSpan={season.months.length}>
                  <span>{season.name}</span>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableHeader>
            <TableRow className="*:border-border hover:bg-transparent [&>:not(:last-child)]:border-r">
              <TableCell></TableCell>
              {monthlyData.map((month) => (
                <TableHead key={month.month} className="h-auto rotate-180 py-3 text-foreground [writing-mode:vertical-lr]">
                  {month.month}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow className="*:border-border [&>:not(:last-child)]:border-r">
              <TableHead className="font-medium text-foreground">Padi</TableHead>
              {monthlyData.map((month, index) => (
                <TableCell
                  key={`${month.month}-${index}`}
                  className={`space-y-1 text-center p-2 ${getStatusColor(month.status, month.hasData)}`}
                  title={month.hasData ? `Curah Hujan: ${month.curah_hujan.toFixed(1)}mm\nKelembapan: ${month.kelembapan.toFixed(1)}%` : "Data tidak tersedia"}
                >
                  <div className="text-xs font-medium">{month.hasData ? month.status : "No Data"}</div>
                  {month.hasData && <div className="text-xs opacity-75">{month.curah_hujan.toFixed(0)}mm</div>}
                </TableCell>
              ))}
            </TableRow>
          </TableBody>
        </Table>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
      <div className="text-sm text-gray-600 mt-4">
        <p>Data yang tersedia: {summaryData?.length || 0} bulan</p>
        <p>Hover pada cell untuk melihat detail curah hujan dan kelembapan</p>
      </div>
      <div>
        <CalendarSeedPlanner />
      </div>
    </div>
  );
}

export { YearlyCalender };
