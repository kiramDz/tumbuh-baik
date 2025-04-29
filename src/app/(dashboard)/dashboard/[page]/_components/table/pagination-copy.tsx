// DataTablePagination.tsx
// import { Table } from "@tanstack/react-table";

// interface DataTablePaginationProps<TData> {
//   table: Table<TData>;
//   pagination?: {
//     currentPage: number;
//     totalPages: number;
//     total: number;
//     onPageChange: (page: number) => void;
//   };
// }



export function DataTablePaginationCopy({ pagination }: { pagination: { currentPage: number; totalPages: number; onPageChange: (page: number) => void } }) {
  return (
    <div className="flex items-center justify-between">
      <button onClick={() => pagination.onPageChange(pagination.currentPage - 1)} disabled={pagination.currentPage === 1}>
        Previous
      </button>
      <span>
        Page {pagination.currentPage} of {pagination.totalPages}
      </span>
      <button onClick={() => pagination.onPageChange(pagination.currentPage + 1)} disabled={pagination.currentPage === pagination.totalPages}>
        Next
      </button>
    </div>
  );
}
