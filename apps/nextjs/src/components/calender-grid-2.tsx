import { useMemo } from "react";
import { format, getMonth } from "date-fns";
import clsx from "clsx";

interface HoltWinterDaily {
  date: string;
  parameters: Record<string, { forecast_value: number }>;
}

interface PlantingCalendarGridProps {
  data: HoltWinterDaily[];
  parameter: string;
  selectedYear: number;
  onYearChange: (year: number) => void;
}

function getColor(value: number) {
  if (value >= 200) return "bg-green-800";
  if (value >= 150) return "bg-green-600";
  if (value >= 100) return "bg-green-400";
  if (value > 0) return "bg-green-200";
  return "bg-neutral-900";
}

function getMonthLabels(dates: string[]) {
  const result: { month: string; span: number }[] = [];
  let prevMonth = "";
  let count = 0;
  for (let i = 0; i < dates.length; i++) {
    const month = format(new Date(dates[i]), "MMM");
    if (month !== prevMonth) {
      if (count > 0) {
        result.push({ month: prevMonth, span: count });
      }
      prevMonth = month;
      count = 1;
    } else {
      count++;
    }
  }
  if (count > 0) {
    result.push({ month: prevMonth, span: count });
  }
  return result;
}

export function PlantingCalendarGrid({ data, parameter }: PlantingCalendarGridProps) {
  const dates = useMemo(() => {
    return data.map((d) => d.date).filter(Boolean);
  }, [data]);

  const monthLabels = getMonthLabels(dates);

  return (
    <div className="overflow-x-auto">
      <div className="flex gap-1 mb-2 ml-[36px]">
        {monthLabels.map((label, idx) => (
          <div key={idx} className="text-xs text-white" style={{ width: `${label.span * 16}px` }}>
            {label.month}
          </div>
        ))}
      </div>
      <div className="flex flex-col gap-1">
        {[...Array(3)].map((_, rowIdx) => (
          <div key={rowIdx} className="flex gap-1">
            {data
              .filter((_, i) => i % 3 === rowIdx)
              .map((item, idx) => {
                const val = item.parameters?.[parameter]?.forecast_value ?? 0;
                return <div key={idx} className={clsx("w-4 h-4 rounded-sm", getColor(val))} title={`${item.date} - ${val.toFixed(1)}`} />;
              })}
          </div>
        ))}
      </div>
    </div>
  );
}
