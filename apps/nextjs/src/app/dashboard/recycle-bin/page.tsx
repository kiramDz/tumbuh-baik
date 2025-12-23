import { Heading } from "@/components/ui/heading";
import { DataTableSkeleton } from "../_components/data-table-skeleton";
import { Suspense } from "react";
import RecycleBinTable from "../_components/recyle-bin";
export const metadata = {
  title: "Dashboard: Recyle Bin",
};

export default async function Page() {
  return (
    <div className="flex flex-col border-none shadow-none space-y-4 p-4 md:px-6">
      <div className="flex items-start justify-start">
        <Heading title="Recycle Bin" description="Manage and restore deleted datasets." />
      </div>
      <Suspense fallback={<DataTableSkeleton columnCount={5} rowCount={8} filterCount={2} />}>
        <RecycleBinTable />
      </Suspense>
    </div>
  );
}