"use client";

import { useState, useMemo } from "react";
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

  const dataItems = data?.items || [];
  const dynamicColumns: ColumnDef<any, any>[] = useMemo(() => {
    if (dataItems.length === 0) return [];

    const flatKeys = new Set<string>();

    const flatten = (obj: any, prefix = "") => {
      for (const key in obj) {
        const val = obj[key];
        const fullKey = prefix ? `${prefix}.${key}` : key;

        if (typeof val === "object" && val !== null && !Array.isArray(val) && !(val instanceof Date)) {
          flatten(val, fullKey);
        } else {
          flatKeys.add(fullKey);
        }
      }
    };

    flatten(dataItems[0]); // gunakan data item pertama

    return Array.from(flatKeys).map((key) => ({
      accessorKey: key,
      header: key,
      cell: ({ row }) => {
        const keys = key.split(".");
        let value = row.original;
        for (const k of keys) {
          value = value?.[k];
        }

        if (value instanceof Date) {
          return value.toISOString().split("T")[0];
        }
        if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}T/.test(value)) {
          return value.split("T")[0];
        }
        return typeof value === "object" ? JSON.stringify(value) : String(value ?? "-");
      },
    }));
  }, [dataItems]);

  if (isLoading) {
    return <DataTableSkeleton columnCount={5} filterCount={1} cellWidths={["10rem", "8rem", "8rem", "8rem", "24rem"]} shrinkZero />;
  }

  if (error) {
    toast.error("Gagal memuat data summary.");
    return <div>Error loading data.</div>;
  }
  return (
    <KaltamTableUI
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
    />
  );
};

export default KaltamTableSummary;
