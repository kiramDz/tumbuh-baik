import { useMemo, useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { parseISO, isWithinInterval, startOfWeek, format, addDays, eachDayOfInterval } from "date-fns";
import { cn } from "@/lib/utils";
import { ThresholdKey, thresholds } from "@/config/tresholds";

interface HoltWinterDaily {
  date: string;
  parameters: Record<ThresholdKey, { forecast_value: number }>;
}

type GridBoxProps = {
  value: number;
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

  return <div className={cn("w-3 h-3 rounded", bgColor)} title={`${parameter}: ${value.toFixed(2)}`} />;
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
    const periods = {
      "KT-1": {
        start: parseISO(`${selectedYear}-09-20`),
        end: parseISO(`${selectedYear + 1}-01-20`),
      },
      "KT-2": {
        start: parseISO(`${selectedYear}-01-21`),
        end: parseISO(`${selectedYear}-06-20`),
      },
      "KT-3": {
        start: parseISO(`${selectedYear}-06-21`),
        end: parseISO(`${selectedYear}-09-19`),
      },
    };

    const result: Record<string, (HoltWinterDaily | null)[]> = {};

    Object.entries(periods).forEach(([key, { start, end }]) => {
      // Filter data untuk periode ini
      const periodData = data.filter((item) => {
        const date = parseISO(item.date);
        return isWithinInterval(date, { start, end });
      });

      // Organisir ke dalam grid mingguan
      result[key] = organizeDataIntoWeeklyGrid(periodData, start, end);
    });

    return result;
  }, [data, selectedYear]);

  const dates = useMemo(() => {
    return data.map((d) => d.date).filter((date): date is string => date !== undefined && date !== null && !isNaN(Date.parse(date)));
  }, [data]);

  function getMonthLabels(dates: string[]) {
    const result: { month: string; span: number }[] = [];
    let prevMonth = "";
    let count = 0;
    for (let i = 0; i < dates.length; i += 7) {
      const date = dates[i];
      if (!date) continue; // Lewati jika tanggal tidak valid
      const month = format(new Date(date), "MMM");
      if (month !== prevMonth) {
        if (count > 0) {
          result.push({ month: prevMonth, span: Math.floor(count / 7) });
        }
        prevMonth = month;
        count = 1;
      } else {
        count++;
      }
    }
    if (count > 0) {
      result.push({ month: prevMonth, span: Math.floor(count / 7) });
    }
    return result;
  }
  // Hitung jumlah kolom (minggu) untuk setiap periode
  const getColumnsCount = (gridData: (HoltWinterDaily | null)[]) => {
    return Math.ceil(gridData.length / 7);
  };
  const monthLabels = getMonthLabels(dates); // Gunakan dates yang sudah difilter

  // Dalam render:

  // Header hari untuk referensi
  const dayLabels = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"];

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

        {(["KT-1", "KT-2", "KT-3"] as const).map((key) => {
          const gridData = grouped[key];
          const columnsCount = getColumnsCount(gridData);

          return (
            <TabsContent key={key} value={key}>
              <div className="mb-4">
                <div className="grid grid-cols-91 gap-1 mb-1">
                  <div className="flex gap-1 mb-2 ml-[36px]">
                    {monthLabels.map(({ month, span }, idx) => (
                      <div key={idx} className={`col-span-${span} text-xs text-center text-black font-medium`}>
                        {month}
                      </div>
                    ))}
                  </div>
                </div>
                {/* Header hari */}
                <div className="flex mb-2">
                  <div className="w-8"></div> {/* Spacer untuk label hari */}
                  {Array.from({ length: columnsCount }, (_, weekIndex) => (
                    <div key={weekIndex} className="w-3 mr-1 text-xs text-center">
                      {/* Bisa tambahkan label minggu di sini jika perlu */}
                    </div>
                  ))}
                </div>

                {/* Grid Calendar */}
                <div className="flex">
                  {/* Label hari di sebelah kiri */}
                  <div className="flex flex-col mr-2">
                    {dayLabels.map((day, index) => (
                      <div key={day} className="h-3 mb-1 text-xs leading-3 text-right w-6">
                        {index % 2 === 1 ? day : ""} {/* Tampilkan label bergantian agar tidak terlalu padat */}
                      </div>
                    ))}
                  </div>

                  {/* Grid kotak-kotak */}
                  <div
                    className="grid gap-1"
                    style={{
                      gridTemplateRows: "repeat(7, 1fr)",
                      gridTemplateColumns: `repeat(${columnsCount}, 1fr)`,
                      gridAutoFlow: "column",
                    }}
                  >
                    {gridData.map((item, index) => {
                      if (!item) {
                        // Kotak kosong untuk tanggal di luar range
                        return <div key={index} className="w-3 h-3 bg-yellow-400 rounded" title="Data hilang" />;
                      }

                      const value = item.parameters[parameter]?.forecast_value ?? 0;
                      return <GridBox key={item.date} parameter={parameter} value={value} date={item.date} />;
                    })}
                  </div>
                </div>
              </div>

              {gridData.length === 0 && <p className="text-center text-gray-500 mt-4">Data tidak tersedia untuk periode ini</p>}
            </TabsContent>
          );
        })}
      </Tabs>
    </div>
  );
}
