import { Heading } from "@/components/ui/heading";
import { Separator } from "@/components/ui/separator";
import { DataTableSkeleton } from "../_components/data-table-skeleton";
import KaltamTableLSTM from "../_components/kaltam/kaltam-table/lstm-tabel";
import { Suspense } from "react";
import { LSTMConfigList } from "../_components/lstmConfig-list";
import { LSTMDialog } from "../_components/lstm-dialog";
import RunLSTMButton from "../_components/lstm-button";
import { LSTMLineChart } from "../_components/chart/lstm-line";
import { LSTMPieChart } from "../_components/chart/lstm-pie";

export const metadata = {
  title: "Dashboard: Kalender Tanam",
};

export default async function Page() {
  return (
    <div className="flex flex-col border-none shadow-none space-y-4 p-4 md:px-6">
      <div className="flex items-start justify-start">
        <Heading title="Kalender Tanam LSTM" description="Manage products (Server side table functionalities.)" />
      </div>
      <Separator />
      <div className="w-full flex gap-4">
        <LSTMDialog />
        <RunLSTMButton />
      </div>
      <LSTMConfigList />
      <Suspense fallback={<DataTableSkeleton columnCount={5} rowCount={8} filterCount={2} />}>
        <KaltamTableLSTM />
      </Suspense>
      <div className="w-full flex flex-col gap-2">
        <LSTMPieChart />
        <LSTMLineChart />
      </div>
    </div>
  );
}
