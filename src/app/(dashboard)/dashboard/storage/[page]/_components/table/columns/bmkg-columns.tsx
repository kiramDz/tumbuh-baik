import { ColumnDef } from "@tanstack/react-table";

import { BmkgDataType } from "@/types/table-schema";
import { format } from "date-fns";

export const bmkgColumns: ColumnDef<BmkgDataType>[] = [
  // {
  //   id: "select",
  //   header: ({ table }) => <Checkbox checked={table.getIsAllPageRowsSelected()} onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)} aria-label="Select all" />,
  //   cell: ({ row }) => <Checkbox checked={row.getIsSelected()} onCheckedChange={(value) => row.toggleSelected(!!value)} aria-label="Select row" />,
  //   enableSorting: false,
  //   enableHiding: false,
  // },
  {
    accessorKey: "timestamp",
    header: "Waktu",
    cell: ({ row }) => {
      const timestamp = row.getValue("timestamp") as string; 
      const date = new Date(timestamp); 
      return format(date, "dd/MM/yyyy HH:mm"); 
    },
  },
  {
    accessorKey: "city",
    header: "Lokasi",
  },
  {
    accessorKey: "temperature",
    header: "Suhu (°C)",
  },
  {
    accessorKey: "humidity",
    header: "Kelembaban (%)",
  },
  {
    accessorKey: "windSpeed",
    header: "Kecepatan Angin (km/h)",
  },
  {
    accessorKey: "lat",
    header: "Lattitude",
  },
  {
    accessorKey: "lon",
    header: "Longitude",
  },
  {
    accessorKey: "pressure",
    header: "Tekanan",
  },
];
