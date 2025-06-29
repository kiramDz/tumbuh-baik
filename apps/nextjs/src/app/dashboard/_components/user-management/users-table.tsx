"use client";

import type { UserType } from "@/types/table-schema";
import { useQueryStates } from "nuqs";
import { useMemo } from "react";
import { DataTable } from "./data-table/data-table";
import { useDataTable } from "@/hooks/use-data-table";
import { usersTableParamsSchema } from "@/server/admin/user/schema";
import type { DataTableFilterField } from "@/types";
import { getColumns } from "./user-table-columns";
import { UsersTableToolbarActions } from "./user-table-toolbar-action";

type UsersTableProps = {
  users: UserType[]; // Directly pass users data instead of promise
  pageCount: number;
};

export function UsersTable({ users, pageCount }: UsersTableProps) {
  const [{ perPage, sort }] = useQueryStates(usersTableParamsSchema);

  const columns = useMemo(() => getColumns(), []);

  const filterFields: DataTableFilterField<UserType>[] = [
    {
      id: "name",
      label: "Name",
      placeholder: "Search by name or email...",
    },
  ];

  const { table } = useDataTable({
    data: users,
    columns,
    pageCount,
    filterFields,
    shallow: false,
    clearOnDefault: true,
    initialState: {
      pagination: { pageIndex: 0, pageSize: perPage },
      sorting: sort,
      columnPinning: { right: ["actions"] },
    },
    getRowId: (originalRow, index) => `${originalRow._id}-${index}`,
  });

  return (
    <DataTable table={table}>
      <UsersTableToolbarActions table={table} />
    </DataTable>
  );
}
