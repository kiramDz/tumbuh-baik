import { ColumnDef } from "@tanstack/react-table";
import { format } from "date-fns";
import { HoltWinterDataType } from "@/types/table-schema";

export const holtWinterColumns: ColumnDef<HoltWinterDataType>[] = [
  {
    accessorKey: "forecast_date",
    header: "Tanggal Prediksi",
    cell: ({ row }) => {
      const val = row.getValue("forecast_date") as string;
      const date = new Date(val);
      return format(date, "dd/MM/yyyy");
    },
  },
  {
    accessorKey: "parameters.RR.forecast_value",
    header: "Curah Hujan (mm)",
  },
  {
    accessorKey: "parameters.RH_AVG.forecast_value",
    header: "Kelembapan Rata-rata (%)",
  },
];
