"use client";

import CalendarSeedPlanner from "./seed-planner";
import { Separator } from "@/components/ui/separator";
import MonthCalendar from "./monthly-calender";
import { PlantingCalendarWrapper } from "../calender-gridWrapper";

// interface HoltWinterDaily {
//   date: string;
//   parameters: Record<ThresholdKey, { forecast_value: number }>;
// }

// function flattenForecastData(data: any): HoltWinterDaily[] {
//   console.log("Raw data received:", data);

//   if (!data) return [];

//   if (Array.isArray(data)) {
//     return data.map((item) => ({
//       date: typeof item.forecast_date === "string" ? item.forecast_date : item.forecast_date?.$date || new Date(item.forecast_date).toISOString(),
//       parameters: {
//         RR_imputed: {
//           forecast_value: item.parameters?.RR_imputed?.forecast_value ?? 0,
//         },
//       } as Record<ThresholdKey, { forecast_value: number }>,
//     }));
//   }

//   const arrayData = data.data || data.results || data.items || [];

//   if (Array.isArray(arrayData)) {
//     return arrayData.map((item) => ({
//       date: typeof item.forecast_date === "string" ? item.forecast_date : item.forecast_date?.$date || new Date(item.forecast_date).toISOString(),
//       parameters: {
//         RR_imputed: {
//           forecast_value: item.parameters?.RR_imputed?.forecast_value ?? 0,
//         },
//       } as Record<ThresholdKey, { forecast_value: number }>,
//     }));
//   }

//   console.warn("Data is not an array:", typeof data, data);
//   return [];
// }

function YearlyCalender() {
  return (
    <div className="spaye-y-8 p-6">
      <MonthCalendar />
      <PlantingCalendarWrapper />
      <Separator className="my-4" />
      <div>
        <CalendarSeedPlanner />
      </div>
    </div>
  );
}

export { YearlyCalender };
