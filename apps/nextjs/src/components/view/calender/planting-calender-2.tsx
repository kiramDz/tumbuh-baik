"use client";

import CalendarSeedPlanner from "./seed-planner";
import { Separator } from "@/components/ui/separator";
import PeriodCalendar from "./monthly-calender-2";
function YearlyCalender() {
  return (
    <div className="spaye-y-6 p-6">
      <PeriodCalendar />
      <Separator className="my-10" />
      <div>
        <CalendarSeedPlanner />
      </div>
    </div>
  );
}

export { YearlyCalender };
