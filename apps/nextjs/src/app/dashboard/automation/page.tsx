import { Metadata } from "next";
import SchedulerStatus from "@/components/scheduler/SchedulerStatus";
import ManualTrigger from "@/components/scheduler/ManualTrigger";
import AutomationConfig from "@/components/scheduler/AutomationConfig";
import ExecutionHistory from "@/components/scheduler/ExecutionHistory";

export const metadata: Metadata = {
  title: "Scheduler Automation | Climate Dashboard",
  description: "Configure and monitor automated climate data processing",
};

export default function AutomationPage() {
  return (
    <div className="container mx-auto p-4 md:p-6 lg:p-8 max-w-7xl space-y-8 animate-in fade-in zoom-in-95 duration-300">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
          Scheduler Automation
        </h1>
        <p className="mt-2 text-gray-500">
          Monitor execution logs, trigger manual processes, and configure
          recurring background tasks for your climate datasets.
        </p>
      </div>

      {/* Top Row: Status + Manual Trigger */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch">
        <div className="lg:col-span-2 flex h-full">
          <div className="w-full">
            <SchedulerStatus />
          </div>
        </div>
        <div className="flex h-full items-start lg:pt-[10%]">
          <div className="w-full">
            <ManualTrigger />
          </div>
        </div>
      </div>

      <hr className="border-gray-200" />

      {/* Configuration Section */}
      <section>
        <AutomationConfig />
      </section>

      <hr className="border-gray-200" />

      {/* Execution History */}
      <section>
        <ExecutionHistory />
      </section>
    </div>
  );
}
