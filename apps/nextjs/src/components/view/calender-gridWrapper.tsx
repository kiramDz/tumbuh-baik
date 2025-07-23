"use client";

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
     const validData = result
       .map((item: any) => ({
         ...item,
         date: item.forecast_date, // Pastikan mapping ke date
       }))
       .filter((item) => item.date && !isNaN(Date.parse(item.date)));
     console.log("Fetched data:", validData); // Debug data
     setData(validData);
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
