"use client";

import LSTMCalendarSeedPlanner from "./lstm_seed";
import { Separator } from "@/components/ui/separator";
import LSTMMonthlyCalendar from "./lstm_monthly_calender";
function LSTMCalender() {
  return (
    <div className="spaye-y-6 p-6">
      <LSTMMonthlyCalendar />
      <Separator className="my-10" />
      <LSTMCalendarSeedPlanner />
    </div>
  );
}

export { LSTMCalender };