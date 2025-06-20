import { ColumnDef } from "@tanstack/react-table";
import { format } from "date-fns";
import { SeedType } from "@/types/table-schema";

export const seedColumns: ColumnDef<SeedType>[] = [
  {
    accessorKey: "name",
    header: "Nama Bibit",
    cell: ({ row }) => row.getValue("name"),
  },
  {
    accessorKey: "duration",
    header: "Durasi (hari)",
    cell: ({ row }) => row.getValue("duration"),
  },
  {
    accessorKey: "createdAt",
    header: "Ditambahkan",
    cell: ({ row }) => {
      const val = row.getValue("createdAt") as string;
      return format(new Date(val), "dd/MM/yyyy");
    },
  },
];
