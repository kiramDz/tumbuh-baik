"use client";

import type { UserType } from "@/types/table-schema";
import type { ColumnDef } from "@tanstack/react-table";
import type { ComponentProps } from "react";
import { UserActions } from "./user-actions";
import { RowCheckbox } from "./row-checkbox";
import { Badge } from "@/components/ui/badge";
import { Note } from "@/components/ui/note";

import { DataTableColumnHeader } from "./data-table/data-tabe-column-header";

export const getColumns = (): ColumnDef<UserType>[] => {
  const roleBadges: Record<"admin" | "user", ComponentProps<typeof Badge>> = {
    admin: {
      variant: "default",
      className: "capitalize",
    },
    user: {
      variant: "outline",
      className: "capitalize",
    },
  };

  return [
    {
      id: "select",
      enableSorting: false,
      enableHiding: false,
      header: ({ table }) => (
        <RowCheckbox
          checked={table.getIsAllPageRowsSelected()}
          ref={(input) => {
            if (input) {
              input.indeterminate = table.getIsSomePageRowsSelected() && !table.getIsAllPageRowsSelected();
            }
          }}
          onChange={(e) => table.toggleAllPageRowsSelected(e.target.checked)}
          aria-label="Select all"
        />
      ),
      cell: ({ row }) => <RowCheckbox checked={row.getIsSelected()} onChange={(e) => row.toggleSelected(e.target.checked)} aria-label="Select row" />,
    },
    {
      accessorKey: "name",
      enableHiding: false,
      size: 160,
      header: ({ column }) => <DataTableColumnHeader column={column} title="Name" />,
      cell: ({ row }) => <Note>{row.getValue("name")}</Note>,
    },
    {
      accessorKey: "email",
      enableSorting: false,
      header: ({ column }) => <DataTableColumnHeader column={column} title="Email" />,
      cell: ({ row }) => <Note>{row.getValue("email")}</Note>,
    },
    {
      accessorKey: "role",
      header: ({ column }) => <DataTableColumnHeader column={column} title="Role" />,
      cell: ({ row }) => {
        const role = row.getValue<"admin" | "user">("role");
        return <Badge {...roleBadges[role]}>{role}</Badge>;
      },
    },
    {
      id: "actions",
      cell: ({ row }) => <UserActions user={row.original} className="float-right" />,
    },
  ];
};
