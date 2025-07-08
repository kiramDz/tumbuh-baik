//act as maintable
"use client";
import { useState } from "react";
import { holtWinterColumns } from "./column";
import { useQuery } from "@tanstack/react-query";
import { getBmkgDaily } from "@/lib/fetch/files.fetch";
import { toast } from "sonner";
import { DataTableSkeleton } from "@/app/dashboard/_components/data-table-skeleton";
import { KaltamTableUI } from "./kaltam-table";
import { exportToCsv } from "@/lib/fetch/files.fetch";

const KaltamTable = () => {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [isExporting, setIsExporting] = useState(false);

  const { data, isLoading, error } = useQuery({
    queryKey: ["bmkg-daily", page, pageSize],
    queryFn: () => getBmkgDaily(page, pageSize),
    refetchOnWindowFocus: false,
  });

  if (isLoading) return <DataTableSkeleton columnCount={7} filterCount={2} cellWidths={["10rem", "30rem", "10rem", "10rem", "6rem", "6rem", "6rem"]} shrinkZero />;

  if (error) {
    toast("Failed to load BMKG data");
    return <div>Error loading data.</div>;
  }

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const result = await exportToCsv(category, sortBy, sortOrder);
      if (result.success) {
        toast.success("Data exported successfully!");
      } else {
        toast.error(result.message || "Failed to export data");
      }
    } catch (error) {
      toast.error("Failed to export data");
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <>
      <KaltamTableUI
        data={data?.items || []}
        columns={holtWinterColumns}
        pagination={{
          currentPage: data?.currentPage || 1,
          totalPages: data?.totalPages || 1,
          total: data?.total || 0,
          pageSize,
          onPageChange: setPage,
          onPageSizeChange: setPageSize,
        }}
        export={{
          onExport: handleExport,
          isExporting,
        }}
      />
    </>
  );
};

export default KaltamTable;
