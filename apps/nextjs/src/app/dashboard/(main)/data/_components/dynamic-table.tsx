"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { getDynamicDatasetData } from "@/lib/fetch/files.fetch";
import { MainTableUI } from "@/app/dashboard/_components/table/main-table-ui";
import { ColumnDef } from "@tanstack/react-table";
import { exportToCsv } from "@/lib/fetch/files.fetch";
interface DynamicMainTableProps {
  collectionName: string; // slug
  columns: string[]; // from dataset_meta
}

export default function DynamicMainTable({ collectionName, columns }: DynamicMainTableProps) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [sortBy, setSortBy] = useState(columns[0] || ""); // default sort by first column
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [isExporting, setIsExporting] = useState(false);

  const { data, isLoading, error } = useQuery({
    queryKey: [collectionName, page, pageSize, sortBy, sortOrder],
    queryFn: () => getDynamicDatasetData(collectionName, page, pageSize, sortBy, sortOrder),
    enabled: !!collectionName && columns.length > 0,
    refetchOnWindowFocus: false,
  });

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const result = await exportToCsv(collectionName, sortBy, sortOrder); // ganti `category` ➜ `collectionName`
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

  // Buat column definition dinamis
  const dynamicColumns: ColumnDef<any, any>[] = columns.map((col) => ({
    accessorKey: col,
    header: col,
    cell: ({ row }) => {
      const value = row.getValue(col);
      return typeof value === "object" ? JSON.stringify(value) : String(value ?? "-");
    },
  }));

  if (isLoading) return <p className="text-sm text-muted-foreground">Memuat data...</p>;
  if (error) return <p className="text-sm text-red-500">Gagal memuat data</p>;

  return (
    <MainTableUI
      data={data?.items || []}
      columns={dynamicColumns}
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
      export={{
        onExport: handleExport,
        isExporting,
      }}
    />
  );
}
