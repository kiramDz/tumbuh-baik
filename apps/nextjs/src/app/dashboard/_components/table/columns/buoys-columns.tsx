// src/app/dashboard/buoys/_components/columns.ts
import { ColumnDef } from "@tanstack/react-table";
import { BuoysDataType } from "@/types/table-schema";
import { format } from "date-fns";

export const buoysColumns: ColumnDef<BuoysDataType>[] = [
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
    accessorKey: "Location",
    header: "Lokasi",
  },
  {
    accessorKey: "SST",
    header: "Suhu Permukaan (°C)",
  },
  {
    // Menggunakan format dengan titik untuk mengakses data
    accessorKey: "TEMP_10.0m",
    header: "10m (°C)",
  },
  {
    accessorKey: "TEMP_60.0m",
    header: "60m (°C)",
  },
  {
    accessorKey: "TEMP_100.0m",
    header: "100m (°C)",
  },
  {
    accessorKey: "TEMP_180.0m",
    header: "180m (°C)",
  },
  {
    accessorKey: "TEMP_300.0m",
    header: "300m (°C)",
  },
  {
    accessorKey: "WSPD",
    header: "Kecepatan Angin (m/s)",
    cell: ({ row }) => {
      const val = row.getValue("WSPD");
      return val === "" ? "-" : val;
    },
  },
  {
    accessorKey: "WDIR",
    header: "Arah Angin (°)",
    cell: ({ row }) => {
      const val = row.getValue("WDIR");
      return val === "" ? "-" : val;
    },
  },
  {
    accessorKey: "RAIN",
    header: "Curah Hujan (mm)",
    cell: ({ row }) => {
      const val = row.getValue("RAIN");
      return val === "" ? "-" : val;
    },
  },
  // Menambahkan kolom untuk RAD dan RH jika diperlukan
  {
    accessorKey: "RAD",
    header: "Radiasi",
    cell: ({ row }) => {
      const val = row.getValue("RAD");
      return val === "" ? "-" : val;
    },
  },
  {
    accessorKey: "RH",
    header: "Kelembaban (%)",
    cell: ({ row }) => {
      const val = row.getValue("RH");
      return val === "" ? "-" : val;
    },
  },
];
