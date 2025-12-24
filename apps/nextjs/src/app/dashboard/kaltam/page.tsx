import { Heading } from "@/components/ui/heading";
import { Separator } from "@/components/ui/separator";
import { DataTableSkeleton } from "../_components/data-table-skeleton";
import KaltamTable from "../_components/kaltam/kaltam-table";
import { Suspense } from "react";
import { ForecastConfigList } from "../_components/forecastConfig-list";
import { ForecastDialog } from "../_components/kaltam-dialog";
import RunForecastButton from "../_components/forecast-button";
import { RainbowGlowGradientLineChart } from "../_components/chart/line-kaltam";
import { RoundedPieChart } from "../_components/chart/pie-kaltam";

export const metadata = {
  title: "Dashboard: Kalender Tanam",
};

export default async function Page() {
  return (
    <div className="flex flex-col border-none shadow-none space-y-4 p-4 md:px-6">
      <div className="flex items-start justify-start">
        <Heading title="Kalender Tanam" description="Kelola prediksi kalender tanam menggunakan model Holt Winter" />
      </div>
      <Separator />
      <div className="w-full flex gap-4">
        <ForecastDialog />
        <RunForecastButton />
      </div>
      <ForecastConfigList />
      <Suspense fallback={<DataTableSkeleton columnCount={5} rowCount={8} filterCount={2} />}>
        <KaltamTable />
      </Suspense>
      <div className="w-full flex flex-col gap-2">
        <RoundedPieChart />
        <RainbowGlowGradientLineChart />
      </div>
    </div>
  );
}
