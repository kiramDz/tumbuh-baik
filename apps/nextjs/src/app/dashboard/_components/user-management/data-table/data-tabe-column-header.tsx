import type { Column } from "@tanstack/react-table";
import type { ComponentProps } from "react";

import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

type DataTableColumnHeaderProps<TData, TValue> = ComponentProps<typeof DropdownMenuTrigger> & {
  column: Column<TData, TValue>;
  title: string;
};

export function DataTableColumnHeader<TData, TValue>({ column, title, className, ...props }: DataTableColumnHeaderProps<TData, TValue>) {
  if (!column.getCanSort() && !column.getCanHide()) {
    return <div className={cn({ className })}>{title}</div>;
  }

  const buttonLabel =
    column.getCanSort() && column.getIsSorted() === "desc" ? "Sorted descending. Click to sort ascending." : column.getIsSorted() === "asc" ? "Sorted ascending. Click to sort descending." : "Not sorted. Click to sort ascending.";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className={cn({ toggleable: true, className })} aria-label={buttonLabel} {...props}>
        {title}
      </DropdownMenuTrigger>

      <DropdownMenuContent align="start">
        {column.getCanSort() && (
          <>
            <DropdownMenuItem aria-label="Sort ascending" onClick={() => column.toggleSorting(false)}>
              Asc
            </DropdownMenuItem>

            <DropdownMenuItem aria-label="Sort descending" onClick={() => column.toggleSorting(true)}>
              Desc
            </DropdownMenuItem>
          </>
        )}
        {column.getCanSort() && column.getCanHide() && <DropdownMenuSeparator />}
        {column.getCanHide() && (
          <DropdownMenuItem aria-label="Hide column" onClick={() => column.toggleVisibility(false)}>
            Hide
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
