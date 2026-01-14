"use client";

import CalendarSeedPlanner from "./seed-planner";
import { Separator } from "@/components/ui/separator";
import PeriodCalendar from "./monthly-calender-2"; // Ubah ke default import

function YearlyCalender() {
  return (
    <div className="space-y-6 p-6"> {/* Perbaiki typo: spaye-y-6 -> space-y-6 */}
      <PeriodCalendar />
      <Separator className="my-10" />
      <CalendarSeedPlanner />
    </div>
  );
}

export { YearlyCalender };
