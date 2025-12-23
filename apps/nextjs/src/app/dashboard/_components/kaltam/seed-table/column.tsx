import { ColumnDef } from "@tanstack/react-table";
import { format } from "date-fns";
import { SeedType } from "@/types/table-schema";
import { SeedActions } from "./seed-actions";

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
  {
    id: "actions",
    header: "Aksi",
    cell: ({ row }) => <SeedActions seed={row.original} />,
  },
];
