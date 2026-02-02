import PageContainer from "@/components/ui/page-container";
import { Heading } from "@/components/ui/heading";
import { Separator } from "@/components/ui/separator";
import { Suspense } from "react";

import { DataTableSkeleton } from "../_components/data-table-skeleton";
import AddSeedDialog from "../_components/kaltam/add-seed";
import SeedTable from "../_components/kaltam/seed-table";
export const metadata = {
  title: "Dashboard: Data Table",
};

export default async function Page() {
  return (
    <>
      <PageContainer scrollable={false}>
        <div className="flex flex-1 flex-col space-y-4">
          <div className="flex items-start justify-start">
            <Heading title="Varietas Padi" description="Manage products (Server side table functionalities.)" />
          </div>
          <Separator />
          <AddSeedDialog />
          <Suspense fallback={<DataTableSkeleton columnCount={5} rowCount={8} filterCount={2} />}>
            <SeedTable />
          </Suspense>
        </div>
      </PageContainer>
    </>
  );
}
