"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getHoltWinterSummary } from "@/lib/fetch/files.fetch";
import { toast } from "sonner";
import { DataTableSkeleton } from "@/app/dashboard/_components/data-table-skeleton";
import { KaltamTableUI } from "../kaltam-table/kaltam-table";
import { ColumnDef } from "@tanstack/react-table";

const KaltamTableSummary = () => {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  const { data, isLoading, error } = useQuery({
    queryKey: ["hw-summary", page, pageSize],
    queryFn: () => getHoltWinterSummary(page, pageSize),
    refetchOnWindowFocus: false,
  });

  if (isLoading) {
    return <DataTableSkeleton columnCount={5} filterCount={1} cellWidths={["10rem", "8rem", "8rem", "8rem", "24rem"]} shrinkZero />;
  }

  if (error) {
    toast.error("Gagal memuat data summary.");
    return <div>Error loading data.</div>;
  }

  const flattenSummaryData = (data: any[]) => {
    return data.map((item) => {
      const rainfall = item.parameters?.RR_imputed?.avg != null ? item.parameters.RR_imputed.avg.toFixed(2) : "-";

      return {
        month: item.month,
        kt_period: item.kt_period,
        status: item.status,
        rainfall_avg: rainfall,
        reason: item.reason,
      };
    });
  };

  const flattenedData = flattenSummaryData(data?.items || []);
  const columns: ColumnDef<any, any>[] = [
    {
      accessorKey: "month",
      header: "Bulan",
      cell: ({ row }) => row.getValue("month"),
    },
    {
      accessorKey: "kt_period",
      header: "KT",
    },
    {
      accessorKey: "status",
      header: "Status",
    },
    {
      accessorKey: "rainfall_avg",
      header: "Curah Hujan (mm/bln)",
    },
    {
      accessorKey: "reason",
      header: "Alasan",
    },
  ];

  return (
    <KaltamTableUI
      data={flattenedData}
      columns={columns}
      pagination={{
        currentPage: data?.currentPage || 1,
        totalPages: data?.totalPages || 1,
        total: data?.total || 0,
        pageSize,
        onPageChange: setPage,
        onPageSizeChange: setPageSize,
      }}
    />
  );
};

export default KaltamTableSummary;
