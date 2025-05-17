import React from "react";

interface DataTableProps {
  data: Record<string, string | number>[];
  columns: string[];
  header?: string[];
  title?: string;
}

const DataTable: React.FC<DataTableProps> = ({ data, columns, header, title }) => {
  const tableHeaders = header || columns;

  return (
    <div className="data-table-container my-6">
      {title && <h3 className="text-xl font-semibold mb-3">{title}</h3>}
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border border-gray-200 rounded-lg">
          <thead className="bg-gray-100">
            <tr>
              {tableHeaders.map((head, index) => (
                <th key={`header-${index}`} className="px-4 py-3 text-left text-sm font-medium text-gray-700 uppercase tracking-wider border-b">
                  {head}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {data.map((row, rowIndex) => (
              <tr key={`row-${rowIndex}`} className={rowIndex % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                {columns.map((col, colIndex) => (
                  <td key={`cell-${rowIndex}-${colIndex}`} className="px-4 py-3 text-sm text-gray-700">
                    {row[col]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DataTable;
