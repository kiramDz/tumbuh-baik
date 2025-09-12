"use client"
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getLSTMDaily, exportLSTMForecastCsv } from "@/lib/fetch/files.fetch";
import { toast } from "sonner";
import { DataTableSkeleton } from "@/app/dashboard/_components/data-table-skeleton";
import { KaltamTableUI } from "./kaltam-table";
import { ColumnDef } from "@tanstack/react-table";

const KaltamTableLSTM = () => {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [isExporting, setIsExporting] = useState(false);

  const { data, isLoading, error } = useQuery({
    queryKey: ["lstm-daily", page, pageSize],
    queryFn: () => getLSTMDaily(page, pageSize),
    refetchOnWindowFocus: false,
  });

  if (isLoading) return <DataTableSkeleton columnCount={7} filterCount={2} cellWidths={["10rem", "30rem", "10rem", "10rem", "6rem", "6rem", "6rem"]} shrinkZero />;

  if (error) {
    toast("Failed to load holt winter data");
    return <div>Error loading data.</div>;
  }
  const flattenForecastData = (data: any[]) => {
    return data.map((item) => {
      const result: Record<string, any> = {
        forecast_date: item.forecast_date,
      };

      for (const [param, val] of Object.entries(item.parameters || {})) {
        // Type guard
        if (val && typeof val === "object" && "forecast_value" in val) {
          result[param] = (val as { forecast_value: number }).forecast_value;
        } else {
          result[param] = "-";
        }
      }

      return result;
    });
  };
  const flattenedData = flattenForecastData(data?.items || []);
  const columns = flattenedData.length ? Object.keys(flattenedData[0]) : [];

  const dynamicColumns: ColumnDef<any, any>[] = columns.map((col) => ({
    accessorKey: col,
    header: col,
    cell: ({ row }) => {
      const value = row.getValue(col);
      if (typeof value === "number") return value.toFixed(2);
      if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}/.test(value)) return value.split("T")[0];
      return value != null ? String(value) : "-";
    },
  }));

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const sortBy = "forecast_date"; // sesuaikan dengan field yang ada di database
      const sortOrder = "desc";

      const result = await exportLSTMForecastCsv(sortBy, sortOrder); // ganti `category` ➜ `collectionName`
      if (result?.success) {
        toast.success("Data exported successfully!");
      } else {
        toast.error(result?.message || "Failed to export data");
      }
    } catch {
      toast.error("Failed to export data");
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <>
      <KaltamTableUI
        data={flattenedData}
        columns={dynamicColumns}
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

export default KaltamTableLSTM;