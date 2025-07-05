import { ColumnDef } from "@tanstack/react-table";

// Type guard untuk cek apakah kolom punya accessorKey
function hasAccessorKey<T>(col: ColumnDef<T, unknown>): col is ColumnDef<T, unknown> & { accessorKey: string } {
  return typeof (col as any).accessorKey === "string";
}

export function extractColumnsFromDef<T>(columnDefs: ColumnDef<T, unknown>[]): Array<{
  key: string;
  header: string;
  hasCustomCell: boolean;
}> {
  return columnDefs.filter(hasAccessorKey).map((col) => ({
    key: col.accessorKey,
    header: typeof col.header === "string" ? col.header : col.accessorKey,
    hasCustomCell: !!col.cell,
  }));
}
