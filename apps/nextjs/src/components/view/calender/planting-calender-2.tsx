import CalendarSeedPlanner from "./seed-planner";
import { Separator } from "@/components/ui/separator";
import SeasonalCalendarTabs from "./season-calender";
import MonthCalendar from "./monthly-calender";

function YearlyCalender() {
  return (
    <div className="spaye-y-8 p-6">
      <MonthCalendar />

      <SeasonalCalendarTabs />
      <Separator className="my-4" />
      <div>
        <CalendarSeedPlanner />
      </div>
    </div>
  );
}

export { YearlyCalender };
