

export function convertToCSV(data: any[], columnInfo: Array<{ key: string; header: string; hasCustomCell: boolean }>): string {
  if (!data.length) return "";

  // Header CSV menggunakan header yang user-friendly
  const headers = columnInfo.map((col) => col.header).join(",");

  // Data rows
  const rows = data.map((item) =>
    columnInfo
      .map((col) => {
        let value = item[col.key];

        // Handle nested keys like "TEMP_10.0m"
        if (col.key.includes(".")) {
          const keys = col.key.split(".");
          value = keys.reduce((obj, key) => obj?.[key], item);
        }

        // Format nilai berdasarkan tipe data
        if (value === null || value === undefined || value === "") {
          return "-";
        }

        // Handle Date formatting
        if (col.key === "Date" && value) {
          const date = new Date(value);
          return date.toLocaleDateString("id-ID");
        }

        // Handle nilai yang mengandung koma, newline, atau quote
        if (typeof value === "string" && (value.includes(",") || value.includes("\n") || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }

        return value;
      })
      .join(",")
  );

  return [headers, ...rows].join("\n");
}
