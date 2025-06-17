import { ColumnDef } from "@tanstack/react-table";

import { BmkgDataType } from "@/types/table-schema";
import { format } from "date-fns";


export const bmkgColumns: ColumnDef<BmkgDataType>[] = [
  {
    accessorKey: "Date",
    header: "Tanggal",
    cell: ({ row }) => {
      const val = row.getValue("Date") as string;
      const date = new Date(val);
      return format(date, "dd/MM/yyyy");
    },
  },
  {
    accessorKey: "TAVG",
    header: "Suhu Rata-rata (°C)",
  },
  {
    accessorKey: "TN",
    header: "Suhu Minimum (°C)",
  },
  {
    accessorKey: "TX",
    header: "Suhu Maksimum (°C)",
  },
  {
    accessorKey: "RH_AVG",
    header: "Kelembaban (%)",
  },
  {
    accessorKey: "RR",
    header: "Curah Hujan (mm)",
  },
  {
    accessorKey: "FF_AVG",
    header: "Kecepatan Angin (km/h)",
  },
  {
    accessorKey: "DDD_X",
    header: "Arah Angin Maks (°)",
  },
  {
    accessorKey: "Season",
    header: "Musim",
  },
];
