import PageContainer from "@/components/ui/page-container";
import { Heading } from "@/components/ui/heading";
import { Separator } from "@/components/ui/separator";
import { DataTableSkeleton } from "../_components/data-table-skeleton";
import KaltamTable from "../_components/kaltam/kaltam-table";
import { Suspense } from "react";

export const metadata = {
  title: "Dashboard: Kalender Tanam",
};

export default async function Page() {
  return (
    <PageContainer scrollable={false}>
      <div className="flex flex-1 flex-col space-y-4">
        <div className="flex items-start justify-start">
          <Heading title="Kalender Tanam" description="Manage products (Server side table functionalities.)" />
        </div>
        <Separator />
        <Suspense fallback={<DataTableSkeleton columnCount={5} rowCount={8} filterCount={2} />}>
          <KaltamTable />
        </Suspense>
      </div>
    </PageContainer>
  );
}
