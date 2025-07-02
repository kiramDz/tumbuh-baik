import { useState } from "react";
import { bmkgColumns } from "./columns/bmkg-columns";
import { buoysColumns } from "./columns/buoys-columns";
import { MainTableUI } from "./main-table-ui";
import { getBmkgData } from "@/lib/fetch/files.fetch";
import { getBuoysData } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";
import { ColumnDef } from "@tanstack/react-table";
import { toast } from "sonner";
import { DataTableSkeleton } from "@/app/dashboard/_components/data-table-skeleton";

interface MainTableProps {
  category: string;
}

type DatasetKey = "bmkg" | "buoys"; // tambahkan jika ada dataset baru

export default function MainTable({ category }: MainTableProps) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [sortBy, setSortBy] = useState("Date");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  const fetchFunction = {
    bmkg: getBmkgData,
    buoys: getBuoysData,
  }[category];

  const columnMap: Record<DatasetKey, ColumnDef<any, any>[]> = {
    bmkg: bmkgColumns,
    buoys: buoysColumns,
  };

  const selectedColumns = columnMap[category as DatasetKey];

  const { data, isLoading, error } = useQuery({
    queryKey: [category, page, pageSize, sortBy, sortOrder], // trigger by all state
    queryFn: () => (fetchFunction ? fetchFunction(page, pageSize, sortBy, sortOrder) : Promise.resolve(null)),
    refetchOnWindowFocus: false,
    enabled: !!fetchFunction,
  });
  
  if (isLoading) return <DataTableSkeleton columnCount={7} filterCount={2} cellWidths={["10rem", "30rem", "10rem", "10rem", "6rem", "6rem", "6rem"]} shrinkZero />;

  if (error) {
    toast("Failed to load BMKG data");
    return <div>Error loading data.</div>;
  }

  return (
    <>
      <MainTableUI
        data={data?.items || []}
        columns={selectedColumns}
        pagination={{
          currentPage: data?.currentPage || 1,
          totalPages: data?.totalPages || 1,
          total: data?.total || 0,
          pageSize,
          onPageChange: setPage,
          onPageSizeChange: setPageSize,
        }}
        sorting={{
          sortBy,
          sortOrder,
          onSortChange: (newSortBy, newSortOrder) => {
            setSortBy(newSortBy);
            setSortOrder(newSortOrder);
          },
        }}
      />
    </>
  );
}
