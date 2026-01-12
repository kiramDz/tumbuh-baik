"use client";
import { useState } from "react";
import { seedColumns } from "./column";
import { useQuery } from "@tanstack/react-query";
import { getSeeds } from "@/lib/fetch/files.fetch";
import { toast } from "sonner";
import { DataTableSkeleton } from "@/app/dashboard/_components/data-table-skeleton";
import { SeedTableUI } from "./seed-table-ui";

const SeedTable = () => {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const { data, isLoading, error } = useQuery({
    queryKey: ["get-seed", page, pageSize],
    queryFn: () => getSeeds(page, pageSize),
    refetchOnWindowFocus: false,
  });

  if (isLoading) return <DataTableSkeleton columnCount={7} filterCount={2} cellWidths={["10rem", "30rem", "10rem", "10rem", "6rem", "6rem", "6rem"]} shrinkZero />;

  if (error) {
    toast("Failed to load Seed data");
    return <div>Error loading data.</div>;
  }
  return (
    <>
      <SeedTableUI
        data={data?.items || []}
        columns={seedColumns}
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

export default SeedTable;
