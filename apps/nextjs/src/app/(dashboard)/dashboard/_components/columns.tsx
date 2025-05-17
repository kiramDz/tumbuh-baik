import { ColumnDef } from "@tanstack/react-table";

import { statuses } from "./filters";
// import { DataTableRowActions } from "./data-table-row-actions";
import { Statustype } from "@/types/table-schema";
import { format } from "date-fns";

export const columns: ColumnDef<Statustype>[] = [
  {
    accessorKey: "source",
    header: "Source",
  },
  {
    accessorKey: "date",
    header: "Waktu",
    cell: ({ row }) => {
      const rawDate = row.getValue("date") as string;
      const date = new Date(rawDate);
      return format(date, "dd/MM/yyyy");
    },
  },

  {
    accessorKey: "record",
    header: "Record",
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: ({ row }) => {
      const status = statuses.find((status) => status.value === row.getValue("status"));

      if (!status) {
        return null;
      }

      return (
        <div className="flex w-[100px] items-center">
          {status.icon && <status.icon className="mr-2 h-4 w-4 text-muted-foreground" />}
          <span>{status.label}</span>
        </div>
      );
    },
    filterFn: (row, id, value) => {
      return value.includes(row.getValue(id));
    },
  },
];
