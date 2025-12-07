"use client";

import * as React from "react";
import {
  ColumnDef,
  ColumnFiltersState,
  VisibilityState,
  flexRender,
  getCoreRowModel,
  getFacetedRowModel,
  getFacetedUniqueValues,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { Download, FileDown, Database } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { DataTablePagination } from "./data-table-pagination";
import { DataTableViewOptions } from "./data-table-view-option";
import { DataTableSort } from "./data-table-sort";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  pagination: {
    currentPage: number;
    totalPages: number;
    total: number;
    pageSize: number;
    onPageChange: (page: number) => void;
    onPageSizeChange: (size: number) => void;
  };
  sorting: {
    sortBy: string;
    sortOrder: "asc" | "desc";
    onSortChange: (sortBy: string, sortOrder: "asc" | "desc") => void;
  };
  export?: {
    onExport: () => void;
    isExporting: boolean;
  };
  isLoading?: boolean;
}

export function MainTableUI<TData, TValue>({
  columns,
  data,
  pagination,
  sorting,
  export: exportProps,
  isLoading,
}: DataTableProps<TData, TValue>) {
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    []
  );
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({});
  const [rowSelection, setRowSelection] = React.useState({});

  const table = useReactTable({
    data,
    columns,
    manualPagination: true,
    manualSorting: true,
    rowCount: pagination.total,
    pageCount: pagination.totalPages,
    onPaginationChange: (updater) => {
      const newPagination =
        typeof updater === "function"
          ? updater(table.getState().pagination)
          : updater;

      if (newPagination.pageSize !== table.getState().pagination.pageSize) {
        pagination.onPageSizeChange(newPagination.pageSize);
      } else {
        pagination.onPageChange(newPagination.pageIndex + 1);
      }
    },
    onSortingChange: (updater) => {
      const newSorting =
        typeof updater === "function"
          ? updater(table.getState().sorting)
          : updater;

      if (newSorting.length > 0 && sorting) {
        const { id, desc } = newSorting[0];
        sorting.onSortChange(id, desc ? "desc" : "asc");
      }
    },
    state: {
      sorting: sorting
        ? [{ id: sorting.sortBy, desc: sorting.sortOrder === "desc" }]
        : [],
      columnVisibility,
      rowSelection,
      columnFilters,
      pagination: {
        pageIndex: pagination.currentPage - 1,
        pageSize: pagination.pageSize || 10,
      },
    },
    enableRowSelection: true,
    onRowSelectionChange: setRowSelection,
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: getFacetedUniqueValues(),
  });

  const { onPageChange, currentPage } = pagination;
  const pageIndex = table.getState().pagination.pageIndex;

  React.useEffect(() => {
    if (pageIndex + 1 !== currentPage) {
      onPageChange(pageIndex + 1);
    }
  }, [pageIndex, currentPage, onPageChange]);

  if (isLoading) {
    return (
      <Card className="border-0 shadow-sm">
        <CardContent className="p-0">
          <div className="flex items-center justify-between p-4 border-b">
            <Skeleton className="h-5 w-32" />
            <div className="flex items-center gap-2">
              <Skeleton className="h-8 w-24" />
              <Skeleton className="h-8 w-8" />
              <Skeleton className="h-8 w-8" />
            </div>
          </div>
          <div className="p-4 space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
          <div className="flex items-center justify-between p-4 border-t">
            <Skeleton className="h-5 w-40" />
            <Skeleton className="h-8 w-64" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-0 shadow-sm overflow-hidden">
      <CardContent className="p-0">
        {/* Toolbar */}
        <div className="flex items-center justify-between p-4 border-b bg-muted/30">
          <p className="text-sm text-muted-foreground">
            {pagination.total.toLocaleString("id-ID")} data ditemukan
          </p>
          <div className="flex items-center gap-2">
            {exportProps && (
              <Button
                variant="outline"
                size="sm"
                onClick={exportProps.onExport}
                disabled={exportProps.isExporting}
                className="h-8 text-xs"
              >
                {exportProps.isExporting ? (
                  <>
                    <FileDown className="w-3.5 h-3.5 mr-1.5 animate-pulse" />
                    Mengunduh...
                  </>
                ) : (
                  <>
                    <Download className="w-3.5 h-3.5 mr-1.5" />
                    Ekspor
                  </>
                )}
              </Button>
            )}
            <DataTableSort
              currentSortOrder={sorting.sortOrder}
              onSortChange={(order) =>
                sorting.onSortChange(sorting.sortBy, order)
              }
            />
            <DataTableViewOptions table={table} />
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id} className="bg-muted/50 hover:bg-muted/50">
                  {headerGroup.headers.map((header) => (
                    <TableHead
                      key={header.id}
                      className="text-xs font-semibold text-foreground whitespace-nowrap"
                    >
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                    </TableHead>
                  ))}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {table.getRowModel().rows?.length ? (
                table.getRowModel().rows.map((row, index) => (
                  <TableRow
                    key={row.id}
                    data-state={row.getIsSelected() && "selected"}
                    className={index % 2 === 0 ? "bg-background" : "bg-muted/20"}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <TableCell
                        key={cell.id}
                        className="text-sm py-3 whitespace-nowrap"
                      >
                        {flexRender(
                          cell.column.columnDef.cell,
                          cell.getContext()
                        )}
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell
                    colSpan={columns.length}
                    className="h-48"
                  >
                    <div className="flex flex-col items-center justify-center text-center">
                      <div className="p-3 rounded-full bg-muted mb-3">
                        <Database className="w-5 h-5 text-muted-foreground" />
                      </div>
                      <p className="font-medium">Tidak ada data</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        Data yang dicari tidak ditemukan
                      </p>
                    </div>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>

        {/* Pagination */}
        <div className="border-t p-4">
          <DataTablePagination
            table={table}
            totalItems={pagination.total}
            onPageChange={pagination.onPageChange}
          />
        </div>
      </CardContent>
    </Card>
  );
}
