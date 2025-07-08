"use client";

import type { UserType } from "@/types/table-schema";
import type { Table } from "@tanstack/react-table";

interface UsersTableToolbarActionsProps {
  table: Table<UserType>;
}
export function UsersTableToolbarActions({ table }: UsersTableToolbarActionsProps) {
  if (!table) return null;
}
