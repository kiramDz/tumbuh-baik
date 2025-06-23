"use client";
import { useState } from "react";
import { userColumns } from "./column";
import { useQuery } from "@tanstack/react-query";
import { getUsers } from "@/lib/fetch/files.fetch";
import { toast } from "sonner";
import { DataTableSkeleton } from "@/app/dashboard/_components/data-table-skeleton";
import { UserTableUI } from "./user-table-ui";
const UserMangementTable = () => {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  const { data, isLoading, error } = useQuery({
    queryKey: ["get-users", page, pageSize],
    queryFn: () => getUsers(page, pageSize),
    refetchOnWindowFocus: false,
  });

  if (isLoading) return <DataTableSkeleton columnCount={5} filterCount={2} cellWidths={["15rem", "20rem", "8rem", "12rem", "12rem"]} shrinkZero />;

  if (error) {
    toast("Failed to load users data");
    return <div>Error loading data.</div>;
  }

  return (
    <>
      <UserTableUI
        data={data?.items || []}
        columns={userColumns}
        pagination={{
          currentPage: data?.currentPage || 1,
          totalPages: data?.totalPages || 1,
          total: data?.total || 0,
          pageSize,
          onPageChange: setPage,
          onPageSizeChange: setPageSize,
        }}
      />
    </>
  );
};

export default UserMangementTable;
