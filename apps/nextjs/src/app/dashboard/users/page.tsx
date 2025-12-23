import PageContainer from "@/components/ui/page-container";
import { Heading } from "@/components/ui/heading";
import { Separator } from "@/components/ui/separator";
import { Suspense } from "react";
import UserMangementTable from "../_components/user-managements/table";
import { DataTableSkeleton } from "../_components/data-table-skeleton";
export const metadata = {
  title: "Dashboard: Data Table",
};

export default async function Page() {
  return (
    <>
      <PageContainer scrollable={false}>
        <div className="flex flex-1 flex-col space-y-4">
          <div className="flex items-start justify-start">
            <Heading title="Kelola Pengguna" description="Manage products (Server side table functionalities.)" />
          </div>
          <Separator />
          <Suspense fallback={<DataTableSkeleton columnCount={5} rowCount={8} filterCount={2} />}>
            <UserMangementTable />
          </Suspense>
        </div>
      </PageContainer>
    </>
  );
}
