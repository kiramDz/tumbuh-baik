'use client'

import { useState, useEffect } from "react";
import { getHoltWinterDailyGrid } from "@/lib/fetch/files.fetch";
import { ThresholdKey } from "@/config/tresholds";
import PlantingCalendarGrid from "../calender-grid";

interface HoltWinterDaily {
  date: string;
  parameters: Record<ThresholdKey, { forecast_value: number }>;
}

export function PlantingCalendarWrapper() {
  const [data, setData] = useState<HoltWinterDaily[]>([]);
  const [selectedYear, setSelectedYear] = useState<number>(2025);

  useEffect(() => {
    async function fetchData() {
      const result = await getHoltWinterDailyGrid();
      setData(
        result.map((item: any) => ({
          ...item,
          date: item.forecast_date, // Petakan forecast_date ke date
        }))
      );
    }
    fetchData();
  }, []);

  return (
    <PlantingCalendarGrid
      data={data}
      parameter="RR_imputed" // Parameter default
      selectedYear={selectedYear}
      onYearChange={setSelectedYear}
    />
  );
}
