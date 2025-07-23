"use client";

import CalendarSeedPlanner from "./seed-planner";
import { Separator } from "@/components/ui/separator";
import MonthCalendar from "./monthly-calender";
import PeriodCalendar from "./monthly-calender-2";
function YearlyCalender() {
  return (
    <div className="spaye-y-8 p-6">
      <MonthCalendar />
      {/* <PlantingCalendarWrapper /> */}
      <PeriodCalendar />
      <Separator className="my-4" />
      <div>
        <CalendarSeedPlanner />
      </div>
    </div>
  );
}

export { YearlyCalender };
