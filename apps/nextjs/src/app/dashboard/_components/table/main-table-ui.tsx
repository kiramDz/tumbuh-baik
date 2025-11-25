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
import { Icons } from "@/app/dashboard/_components/icons";
import { Loader2, Download } from "lucide-react";
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
import { useState } from "react";
import { DataTableSort } from "./data-table-sort";
import { Button } from "@/components/ui/button";
import toast from "react-hot-toast";
import PreprocessingModal from "../../(main)/data/_components/preprocessing/preprocessing-modal";

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
  // Add preprocessing props
  preprocessing?: {
    collectionName: string;
    isNasaDataset: boolean;
    isBmkgDataset: boolean;
    isAPI?: boolean;
    onPreprocessingComplete?: () => void;
  };
}

export function MainTableUI<TData, TValue>({
  columns,
  data,
  pagination,
  sorting,
  export: exportProps,
  preprocessing,
}: DataTableProps<TData, TValue>) {
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    []
  );
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({});
  const [rowSelection, setRowSelection] = React.useState({});

  // Add preprocessing modal state
  const [isPreprocessingModalOpen, setIsPreprocessingModalOpen] =
    useState(false);

  const canPreprocess =
    preprocessing?.isNasaDataset || preprocessing?.isBmkgDataset;

  const getPreprocessingButtonText = () => {
    if (preprocessing?.isNasaDataset) return "Preprocess NASA";
    if (preprocessing?.isBmkgDataset) return "Preprocess BMKG";
  };

  const table = useReactTable({
    data,
    columns,
    manualPagination: true,
    manualSorting: true, // PENTING: Enable manual sorting
    rowCount: pagination.total,
    pageCount: pagination.totalPages,
    onPaginationChange: (updater) => {
      const newPagination =
        typeof updater === "function"
          ? updater(table.getState().pagination)
          : updater;

      // Trigger perubahan page dan pageSize
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

  // Handle page changes
  const { onPageChange, currentPage } = pagination;
  const pageIndex = table.getState().pagination.pageIndex;

  React.useEffect(() => {
    if (pageIndex + 1 !== currentPage) {
      onPageChange(pageIndex + 1);
    }
  }, [pageIndex, currentPage, onPageChange]);

  // Add preprocessing modal handlers
  const handlePreprocessingClick = () => {
    setIsPreprocessingModalOpen(true);
  };

  const handlePreprocessingSuccess = (result: any) => {
    toast.success(
      `🎉 Preprocessing Completed! Successfully processed ${result.recordCount?.toLocaleString()} records. Quality: ${
        result.preprocessing_report?.quality_metrics?.completeness_percentage ||
        100
      }%`,
      {
        duration: 5000,
        position: "bottom-right",
        style: {
          background: "#10B981",
          color: "#fff",
        },
      }
    );

    setTimeout(() => {
      setIsPreprocessingModalOpen(false);

      if (preprocessing?.onPreprocessingComplete) {
        preprocessing.onPreprocessingComplete();
      }
    }, 2000);
  };
  return (
    <>
      <div className="space-y-4 ">
        <div className="w-full flex items-center gap-2 justify-end">
          {/* Add Preprocessing Button condition */}
          {canPreprocess && (
            <Button
              variant="default"
              size="sm"
              onClick={handlePreprocessingClick}
              className="h-8 bg-green-600 hover:bg-green-700 text-white"
            >
              <Icons.play className="mr-2 h-4 w-4" />
              {getPreprocessingButtonText()}
            </Button>
          )}

          {/* {preprocessing?.isNasaDataset && (
            <Button
              variant="default"
              size="sm"
              onClick={handlePreprocessingClick}
              className="h-8 bg-green-600 hover:bg-green-700 text-white"
            >
              <Icons.play className="mr-2 h-4 w-4" />
              Preprocess Data
            </Button>
          )} */}
          {exportProps && (
            <Button
              variant="outline"
              size="sm"
              onClick={exportProps.onExport}
              disabled={exportProps.isExporting}
              className="h-8"
            >
              {exportProps.isExporting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Exporting...
                </>
              ) : (
                <>
                  <Download className="mr-2 h-4 w-4" />
                  Export CSV
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
        <div className="rounded-md border">
          <Table className="rounded-md">
            <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => {
                    return (
                      <TableHead key={header.id}>
                        {header.isPlaceholder
                          ? null
                          : flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                            )}
                      </TableHead>
                    );
                  })}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {table.getRowModel().rows?.length ? (
                table.getRowModel().rows.map((row) => (
                  <TableRow
                    key={row.id}
                    data-state={row.getIsSelected() && "selected"}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <TableCell key={cell.id}>
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
                    className="h-24 text-center"
                  >
                    No results.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
        <DataTablePagination
          table={table}
          totalItems={pagination.total}
          onPageChange={pagination.onPageChange}
        />
      </div>
      {canPreprocess && (
        <PreprocessingModal
          collectionName={preprocessing.collectionName}
          isNasaDataset={preprocessing.isNasaDataset}
          isBmkgDataset={preprocessing.isBmkgDataset}
          isAPI={preprocessing.isAPI}
          isOpen={isPreprocessingModalOpen}
          onClose={() => setIsPreprocessingModalOpen(false)}
          onSuccess={handlePreprocessingSuccess}
        />
      )}
      {/* {preprocessing?.isNasaDataset && (
        <PreprocessingModal
          collectionName={preprocessing.collectionName}
          isOpen={isPreprocessingModalOpen}
          onClose={() => setIsPreprocessingModalOpen(false)}
          onSuccess={handlePreprocessingSuccess}
        />
      )} */}
    </>
  );
}
