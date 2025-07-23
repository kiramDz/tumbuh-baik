import { useMemo, useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { parseISO, isWithinInterval, startOfWeek, format, addDays, eachDayOfInterval } from "date-fns";
import { cn } from "@/lib/utils";
import { ThresholdKey, thresholds } from "@/config/tresholds";
// import { EvaluateGridColor } from "@/lib/evaluate-gridColor";

// Type untuk data harian Holt-Winters
interface HoltWinterDaily {
  date: string;
  parameters: Record<ThresholdKey, { forecast_value: number }>;
}

type GridBoxProps = {
  value: number; // forecast_value dari RR_imputed
  parameter: string; // misalnya "RR_imputed"
};

const generateDateRange = (start: string, end: string): HoltWinterDaily[] => {
  const startDate = parseISO(start);
  const endDate = parseISO(end);
  return eachDayOfInterval({ start: startDate, end: endDate }).map((date) => ({
    date: date.toISOString(),
    parameters: {} as Record<ThresholdKey, { forecast_value: number }>,
  }));
};

const fillMissingDates = (data: HoltWinterDaily[], period: "KT-1" | "KT-2" | "KT-3", year: number): HoltWinterDaily[] => {
  let start: string, end: string;
  if (period === "KT-1") {
    start = `${year}-09-20`;
    end = `${year + 1}-01-20`;
  } else if (period === "KT-2") {
    start = `${year}-01-21`;
    end = `${year}-06-20`;
  } else {
    start = `${year}-06-21`;
    end = `${year}-09-19`;
  }

  const dateRange = generateDateRange(start, end);
  const dataMap = new Map(data.map((item) => [item.date, item]));

  return dateRange.map((item) => dataMap.get(item.date) || item);
};

const evaluateSingleGridColor = (parameter: string, value: number): "gray" | "green" | "red" => {
  const thresholdKey = parameter as ThresholdKey;
  const threshold = thresholds[thresholdKey];

  if (!threshold) return "gray";

  const { min, max } = threshold;

  if (value < min) {
    return "gray";
  } else if (value > max) {
    return "red";
  } else {
    return "green";
  }
};

const GridBox = ({ value, parameter }: GridBoxProps) => {
  const colorKey = evaluateSingleGridColor(parameter, value);

  const bgColor = {
    gray: "bg-gray-300",
    green: "bg-green-500",
    red: "bg-red-500",
  }[colorKey];

  return <div className={cn("w-6 h-6 rounded", bgColor)} title={`${parameter}: ${value.toFixed(2)}`} />;
};

interface CalendarGridProps {
  data: HoltWinterDaily[];
  parameter: ThresholdKey;
  selectedYear: number; // Tambahkan prop untuk tahun
  onYearChange: (year: number) => void; // Callback untuk perubahan tahun
}
function organizeDataIntoWeeklyGrid(data: HoltWinterDaily[], startDate: Date, endDate: Date): (HoltWinterDaily | null)[] {
  // Buat map dari tanggal ke data
  const dataMap = new Map<string, HoltWinterDaily>();
  data.forEach((item) => {
    dataMap.set(item.date.split("T")[0], item); // Gunakan format YYYY-MM-DD
  });

  // Mulai dari Senin minggu pertama
  const firstMonday = startOfWeek(startDate, { weekStartsOn: 1 });

  const gridData: (HoltWinterDaily | null)[] = [];
  let currentDate = firstMonday;

  // Loop sampai melewati end date
  while (currentDate <= endDate || gridData.length % 7 !== 0) {
    const dateStr = format(currentDate, "yyyy-MM-dd");
    const dataForDate = dataMap.get(dateStr);

    if (currentDate >= startDate && currentDate <= endDate) {
      gridData.push(dataForDate || null);
    } else {
      gridData.push(null); // Hari di luar range tapi perlu untuk melengkapi grid
    }

    currentDate = addDays(currentDate, 1);
  }

  return gridData;
}

export default function PlantingCalendarGrid({ data, parameter, selectedYear, onYearChange }: CalendarGridProps) {
  const [tab, setTab] = useState<"KT-1" | "KT-2" | "KT-3">("KT-1");

  const grouped = useMemo(() => {
    const map: Record<string, HoltWinterDaily[]> = {
      "KT-1": [],
      "KT-2": [],
      "KT-3": [],
    };

    data.forEach((item) => {
      const date = parseISO(item.date);
      const year = date.getFullYear();
      if (year !== selectedYear) return;

      console.log(`Processing item: ${item.date}, year: ${year}, parameters:`, item.parameters);

      if (
        isWithinInterval(date, {
          start: parseISO(`${selectedYear}-09-20`),
          end: parseISO(`${selectedYear + 1}-01-20`),
        })
      ) {
        map["KT-1"].push(item);
      } else if (
        isWithinInterval(date, {
          start: parseISO(`${selectedYear}-01-21`),
          end: parseISO(`${selectedYear}-06-20`),
        })
      ) {
        map["KT-2"].push(item);
      } else if (
        isWithinInterval(date, {
          start: parseISO(`${selectedYear}-06-21`),
          end: parseISO(`${selectedYear}-09-19`),
        })
      ) {
        map["KT-3"].push(item);
      }
    });

    console.log("Grouped data:", {
      "KT-1": map["KT-1"].length,
      "KT-2": map["KT-2"].length,
      "KT-3": map["KT-3"].length,
    });

    return map;
  }, [data, selectedYear]);

  // Hitung jumlah minggu untuk setiap periode
  const getWeekCount = (period: "KT-1" | "KT-2" | "KT-3") => {
    const dates = grouped[period].map((item) => parseISO(item.date));
    if (dates.length === 0) return 0;
    const start = dates[0]; // Hapus .date, gunakan langsung objek Date dari parseISO
    const end = dates[dates.length - 1];
    const diffDays = Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
    return Math.ceil(diffDays / 7);
  };

  const daysOfWeek = ["Mon", "Wed", "Fri"]; // Label hari (contoh: Senin, Rabu, Jumat)

  return (
    <div>
      <select value={selectedYear} onChange={(e) => onYearChange(Number(e.target.value))} className="mb-4 p-2 border rounded">
        <option value={2025}>Periode 2025</option>
        <option value={2026}>Periode 2026</option>
        <option value={2027}>Periode 2027</option>
      </select>
      <Tabs defaultValue="KT-1" value={tab} onValueChange={(v) => setTab(v as any)}>
        <TabsList className="mb-4">
          <TabsTrigger value="KT-1">KT-1</TabsTrigger>
          <TabsTrigger value="KT-2">KT-2</TabsTrigger>
          <TabsTrigger value="KT-3">KT-3</TabsTrigger>
        </TabsList>

        {(["KT-1", "KT-2", "KT-3"] as const).map((key) => (
          <TabsContent key={key} value={key}>
            <div className="flex">
              {/* Label Hari (kiri) */}
              <div className="mr-2">
                {daysOfWeek.map((day, index) => (
                  <div key={index} className="text-center mb-1 text-gray-500">
                    {day}
                  </div>
                ))}
              </div>
              {/* Grid Utama */}
              <div
                className="grid"
                style={{
                  gridTemplateColumns: `repeat(${getWeekCount(key)}, minmax(10px, 1fr))`,
                  gap: "1px",
                }}
              >
                {/* Label Bulan (atas) - Placeholder, perlu dihitung berdasarkan tanggal */}
                <div className="flex mb-1">
                  {grouped[key]
                    .reduce((months, item) => {
                      const month = parseISO(item.date).toLocaleString("en-US", { month: "short" });
                      if (!months.includes(month)) months.push(month);
                      return months;
                    }, [] as string[])
                    .map((month, index) => (
                      <div key={index} className="text-center text-gray-500 mx-2">
                        {month}
                      </div>
                    ))}
                </div>
                {grouped[key].map((item, index) => {
                  const date = parseISO(item.date);
                  const weekIndex = Math.floor(index / 7); // Indeks minggu
                  const dayIndex = index % 7; // Indeks hari dalam minggu
                  if (dayIndex === 0 || dayIndex === 2 || dayIndex === 4) {
                    // Hanya tampilkan Mon, Wed, Fri
                    return <GridBox key={item.date} parameter={parameter} value={item.parameters[parameter]?.forecast_value ?? 0} />;
                  }
                  return null;
                })}
              </div>
            </div>
            {grouped[key].length === 0 && <p className="text-center text-gray-500 mt-4">Data tidak tersedia untuk periode ini</p>}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
