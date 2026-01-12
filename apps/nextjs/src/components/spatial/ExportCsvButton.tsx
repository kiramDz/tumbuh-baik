"use client";

import { useState } from "react";
import { exportSpatialAnalysisCsv } from "@/lib/fetch/spatial.map.fetch";
import type { SpatialAnalysisResponse } from "@/lib/fetch/spatial.map.fetch";

interface ExportCsvButtonProps {
  data: SpatialAnalysisResponse | null;
  isLoading?: boolean;
  disabled?: boolean;
  className?: string;
  variant?: "primary" | "secondary";
}

export function ExportCsvButton({
  data,
  isLoading = false,
  disabled = false,
  className = "",
  variant = "primary",
}: ExportCsvButtonProps) {
  const [isExporting, setIsExporting] = useState(false);
  const [exportStatus, setExportStatus] = useState<{
    type: "success" | "error" | null;
    message: string;
  }>({ type: null, message: "" });

  const handleExport = async () => {
    if (!data || isExporting) return;
    setIsExporting(true);
    setExportStatus({ type: null, message: "" });

    try {
      const timestamp = new Date().toISOString().split("T")[0];
      const filename = `rice_suitability_analysis_${timestamp}.csv`;
      const result = await exportSpatialAnalysisCsv(data, filename);

      if (result.success) {
        setExportStatus({
          type: "success",
          message: `Successfully exported ${data.features.length} districts`,
        });
        // Clear success message after 3 seconds
        setTimeout(() => {
          setExportStatus({ type: null, message: "" });
        }, 3000);
      } else {
        throw new Error(result.message);
      }
    } catch (error: any) {
      console.error("Export failed:", error);
      setExportStatus({
        type: "error",
        message: `Export failed: ${error.message || "Unknown error"}`,
      });
      setTimeout(() => {
        setExportStatus({ type: null, message: "" });
      }, 5000);
    } finally {
      setIsExporting(false);
    }
  };
  const isDisabled = !data || isLoading || disabled || isExporting;
  const buttonStyles = {
    primary: "bg-blue-600 hover:bg-blue-700 text-white",
    secondary: "bg-gray-200 hover:bg-gray-300 text-gray-800",
  };
  return (
    <div className={`space-y-2 ${className}`}>
      <button
        onClick={handleExport}
        disabled={isDisabled}
        className={`
          w-full py-2 px-4 rounded-lg font-medium transition-colors
          ${buttonStyles[variant]}
          ${isDisabled ? "opacity-50 cursor-not-allowed" : "hover:shadow-md"}
        `}
      >
        {isExporting ? (
          <div className="flex items-center justify-center space-x-2">
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            <span>Exporting...</span>
          </div>
        ) : (
          <div className="flex items-center justify-center space-x-2">
            <span>📊</span>
            <span>Export CSV</span>
          </div>
        )}
      </button>

      {/* Status Messages */}
      {exportStatus.type && (
        <div
          className={`
          p-2 rounded text-sm text-center
          ${
            exportStatus.type === "success"
              ? "bg-green-50 text-green-700 border border-green-200"
              : "bg-red-50 text-red-700 border border-red-200"
          }
        `}
        >
          {exportStatus.type === "success" ? "✅" : "❌"} {exportStatus.message}
        </div>
      )}

      {/* Helper Text */}
      <p className="text-xs text-gray-500 text-center">
        {data
          ? `${data.features?.length || 0} districts ready for export`
          : "Run analysis first to enable export"}
      </p>
    </div>
  );
}
