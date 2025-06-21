import PageContainer from "@/components/ui/page-container";
import { Heading } from "@/components/ui/heading";
import { Separator } from "@/components/ui/separator";
import { DataTableSkeleton } from "../_components/data-table-skeleton";
import { Suspense } from "react";
import { withAdminPage } from "../_components/auth-hoc";

export const metadata = {
  title: "Dashboard: User Mangement",
};

type UsersPageProps = {
  searchParams: Record<string, string | string[] | undefined>;
};

const UsersPage = async ({ searchParams }: UsersPageProps) => {
  const search = {
    q: searchParams.q ?? "",
    page: Number(searchParams.page ?? 1),
    // dst...
  };

  const usersPromise = findUsers(search);

  return (
    <>
      <PageContainer scrollable={false}>
        <div className="flex flex-1 flex-col space-y-4">
          <div className="flex items-start justify-start">
            <Heading title="Kalender Tanam" description="Manage products (Server side table functionalities.)" />
          </div>
          <Separator />
          <Suspense fallback={<DataTableSkeleton columnCount={5} rowCount={8} filterCount={2} />}></Suspense>
        </div>
      </PageContainer>
    </>
  );
};

export default withAdminPage(UsersPage); //pastiakan hanya user dengan role admin yg bs masuk keisini 
