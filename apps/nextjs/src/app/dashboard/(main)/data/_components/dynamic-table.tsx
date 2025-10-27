"use client";

import { useState } from "react";
import { getDynamicDatasetData } from "@/lib/fetch/files.fetch";
import { MainTableUI } from "@/app/dashboard/_components/table/main-table-ui";
import { ColumnDef } from "@tanstack/react-table";
import { exportDatasetCsv } from "@/lib/fetch/files.fetch";
import { Button } from "@/components/ui/button";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import RefreshSingleDatasetDialog from "@/app/dashboard/(main)/data/_components/refresh-single-dataset-dialog";
import toast from "react-hot-toast";

interface DynamicMainTableProps {
  collectionName: string; // slug
  columns: string[]; // from dataset_meta
  datasetId?: string;
  datasetName: string;
}

export default function DynamicMainTable({
  collectionName,
  columns,
  datasetId,
  datasetName,
}: DynamicMainTableProps) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [sortBy, setSortBy] = useState(columns[0] || ""); // default sort by first column
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [isExporting, setIsExporting] = useState(false);
  const [showRefreshDialog, setShowRefreshDialog] = useState(false);
  const queryClient = useQueryClient();

  const isNasaDataset =
    datasetName.toLowerCase().includes("nasa") ||
    datasetName.toLowerCase().includes("power");

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: [collectionName, page, pageSize, sortBy, sortOrder],
    queryFn: () =>
      getDynamicDatasetData(collectionName, page, pageSize, sortBy, sortOrder),
    enabled: !!collectionName && columns.length > 0,
    refetchOnWindowFocus: false,
  });

  const handleExport = async () => {
    setIsExporting(true);
    try {
      // Always export in ascending order (oldest to newest)
      const result = await exportDatasetCsv(collectionName, sortBy, "asc");
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

  // Handle refresh completion - invalidate all related queries and refresh
  const handleRefreshComplete = async () => {
    try {
      // Show loading toast
      const loadingToast = toast.loading("Memperbarui data tabel...", {
        position: "bottom-right",
      });

      // Invalidate all queries related to this collection
      await queryClient.invalidateQueries({
        queryKey: [collectionName],
      });

      // Force refetch the current data
      await refetch();

      // Invalidate the main datasets query to update the dataset list
      await queryClient.invalidateQueries({
        queryKey: ["datasets"],
      });

      // Dismiss loading toast and show success
      toast.dismiss(loadingToast);
      toast.success("Data tabel berhasil diperbarui!", {
        duration: 3000,
        position: "bottom-right",
      });
    } catch (error) {
      toast.error("Gagal memperbarui data tabel");
    }
  };

  // Buat column definition dinamis
  const dynamicColumns: ColumnDef<any, any>[] = columns.map((col) => ({
    accessorKey: col,
    header: col,
    cell: ({ row }) => {
      const value = row.getValue(col);
      if (value instanceof Date) {
        return value.toISOString().split("T")[0]; // Format jadi YYYY-MM-DD
      }
      if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}T/.test(value)) {
        return value.split("T")[0]; // Format string ISO → ambil bagian tanggal
      }
      return typeof value === "object"
        ? JSON.stringify(value)
        : String(value ?? "-");
    },
  }));

  const preprocessingProps = {
    collectionName,
    isNasaDataset,
    onPreprocessingComplete: async () => {
      try {
        const loadingToast = toast.loading(
          "Memperbarui data setelah preprocessing...",
          {
            position: "bottom-right",
          }
        );

        await queryClient.invalidateQueries({ queryKey: [collectionName] });
        await refetch();

        toast.dismiss(loadingToast);
        toast.success("Data tabel diperbarui setelah preprocessing!", {
          duration: 3000,
          position: "bottom-right",
        });
      } catch (error) {
        toast.error("Gagal memperbarui data setelah preprocessing");
      }
    },
  };

  if (isLoading)
    return <p className="text-sm text-muted-foreground">Memuat data...</p>;
  if (error) return <p className="text-sm text-red-500">Gagal memuat data</p>;

  return (
    <>
      <div className="flex justify-end mb-2">
        <Button variant="outline" onClick={() => setShowRefreshDialog(true)}>
          Refresh Dataset
        </Button>
        <RefreshSingleDatasetDialog
          datasetId={datasetId || ""}
          datasetName={datasetName || collectionName}
          open={showRefreshDialog}
          onOpenChange={setShowRefreshDialog}
          onRefreshComplete={handleRefreshComplete}
        />
      </div>
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
        // Prerpocessing props
        preprocessing={preprocessingProps}
      />
    </>
  );
}
