//act as maintable
"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getHoltWinterDaily, exportHoltWinterCsv } from "@/lib/fetch/files.fetch";
import { toast } from "sonner";
import { DataTableSkeleton } from "@/app/dashboard/_components/data-table-skeleton";
import { KaltamTableUI } from "./kaltam-table";
import { ColumnDef } from "@tanstack/react-table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, TrendingUp } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

const KaltamTable = () => {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [isExporting, setIsExporting] = useState(false);

  const { data, isLoading, error } = useQuery({
    queryKey: ["hw-daily", page, pageSize],
    queryFn: () => getHoltWinterDaily(page, pageSize),
    refetchOnWindowFocus: false,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-lg">Hasil Peramalan Holt-Winters</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <DataTableSkeleton 
            columnCount={7} 
            filterCount={2} 
            cellWidths={["10rem", "30rem", "10rem", "10rem", "6rem", "6rem", "6rem"]} 
            shrinkZero 
          />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-lg">Hasil Peramalan Holt-Winters</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Gagal memuat data peramalan. Silakan coba lagi.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
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
    header: col === "forecast_date" ? "Tanggal" : col.toUpperCase(),
    cell: ({ row }) => {
      const value = row.getValue(col);
      
      if (col === "forecast_date") {
        if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}/.test(value)) {
          return (
            <span className="font-medium">
              {new Date(value).toLocaleDateString("id-ID", {
                day: "numeric",
                month: "short",
                year: "numeric"
              })}
            </span>
          );
        }
      }
      
      if (typeof value === "number") {
        return <span className="tabular-nums">{value.toFixed(2)}</span>;
      }
      
      return value != null ? String(value) : "-";
    },
  }));

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const sortBy = "forecast_date";
      const sortOrder = "asc";

      const result = await exportHoltWinterCsv(sortBy, sortOrder);
      if (result?.success) {
        toast.success("Data berhasil diekspor");
      } else {
        toast.error(result?.message || "Gagal mengekspor data");
      }
    } catch {
      toast.error("Gagal mengekspor data");
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-lg">Hasil Peramalan Holt-Winters</CardTitle>
          </div>
          <Badge variant="secondary" className="font-normal">
            {data?.total || 0} data
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
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
      </CardContent>
    </Card>
  );
};

export default KaltamTable;
