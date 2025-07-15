import { Heading } from "@/components/ui/heading";
import { Separator } from "@/components/ui/separator";
import { DataTableSkeleton } from "../_components/data-table-skeleton";
import KaltamTable from "../_components/kaltam/kaltam-table";
import KaltamTableSummary from "../_components/kaltam/kaltam-table-summary";
import { Suspense } from "react";
import { ForecastConfigList } from "../_components/forecastConfig-list";
import { ForecastDialog } from "../_components/kaltam-dialog";
import RunForecastButton from "../_components/forecast-button";

export const metadata = {
  title: "Dashboard: Kalender Tanam",
};

export default async function Page() {
  return (
    <div className="flex flex-col space-y-4 p-4 md:px-6">
      <div className="flex items-start justify-start">
        <Heading title="Kalender Tanam" description="Manage products (Server side table functionalities.)" />
      </div>
      <Separator />
      <div className="w-full flex gap-4">
        <ForecastDialog />
        <RunForecastButton />
      </div>
      <ForecastConfigList />
      <Suspense fallback={<DataTableSkeleton columnCount={5} rowCount={8} filterCount={2} />}>
        <KaltamTable />
        <KaltamTableSummary />
      </Suspense>
    </div>
  );
}
