"use client";
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { GetRecycleBinDatasets, RestoreDataset, PermanentDeleteDataset } from "@/lib/fetch/files.fetch";
import { toast } from "sonner";
import { DataTableSkeleton } from "@/app/dashboard/_components/data-table-skeleton";
import { RecycleBinUI } from "./recyle-bin-table";
import { ColumnDef } from "@tanstack/react-table";
import { RecycleBinType } from "@/types/table-schema";
import { Ellipsis } from "lucide-react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

const RecycleBinTable = () => {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ["recycle-bin", page, pageSize],
    queryFn: () => GetRecycleBinDatasets(),
    refetchOnWindowFocus: false,
  });

  if (isLoading) return <DataTableSkeleton columnCount={4} filterCount={2} cellWidths={["10rem", "30rem", "10rem", "10rem"]} shrinkZero />;

  if (error) {
    toast("Failed to load recycle bin data");
    return <div>Error loading data.</div>;
  }

  const handleRestore = async (collectionName: string) => {
    try {
      await RestoreDataset(collectionName);
      toast.success("Dataset restored successfully!");
      queryClient.invalidateQueries({ queryKey: ["recycle-bin"] });
      queryClient.invalidateQueries({ queryKey: ["datasets"] }); // refresh main datasets too
    } catch (error) {
      console.error("Failed to restore dataset:", error);
      toast.error("Failed to restore dataset");
    }
  };

  const handlePermanentDelete = async (collectionName: string) => {
    try {
      await PermanentDeleteDataset(collectionName);
      toast.success("Dataset permanently deleted!");
      queryClient.invalidateQueries({ queryKey: ["recycle-bin"] });
    } catch (error) {
      console.error("Failed to permanently delete dataset:", error);
      toast.error("Failed to permanently delete dataset");
    }
  };
  const recycleBinColumns: ColumnDef<RecycleBinType>[] = [
    {
      accessorKey: "name",
      header: "Dataset Name",
      cell: ({ row }) => row.getValue("name"),
    },
    {
      accessorKey: "description",
      header: "Description",
      cell: ({ row }) => row.getValue("description"),
    },
    {
      id: "actions",
      header: "Actions",
      cell: ({ row }) => (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="p-2 hover:bg-gray-100 rounded">
              <Ellipsis className="h-4 w-4" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => handleRestore(row.original.collectionName)}>Restore</DropdownMenuItem>
            <DropdownMenuItem onClick={() => handlePermanentDelete(row.original.collectionName)} className="text-red-600">
              Delete Permanently
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ),
    },
  ];
  return (
    <RecycleBinUI
      data={data?.items || []}
      columns={recycleBinColumns}
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

export default RecycleBinTable;