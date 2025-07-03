export function convertToCSV(data: any[], columns: string[]): string {
  if (!data.length) return "";

  // Header CSV
  const headers = columns.join(",");

  // Data rows
  const rows = data.map((item) =>
    columns
      .map((col) => {
        const value = item[col];
        // Handle nilai yang mengandung koma, newline, atau quote
        if (typeof value === "string" && (value.includes(",") || value.includes("\n") || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value ?? "";
      })
      .join(",")
  );

  return [headers, ...rows].join("\n");
}
