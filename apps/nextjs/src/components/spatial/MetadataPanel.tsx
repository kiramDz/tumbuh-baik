"use client";
import type { SpatialAnalysisResponse } from "@/lib/fetch/spatial.map.fetch";

interface MetadataPanelProps {
  metadata: SpatialAnalysisResponse["metadata"] | null;
  isLoading?: boolean;
  className?: string;
}
export function MetadataPanel({
  metadata,
  isLoading = false,
  className = "",
}: MetadataPanelProps) {
  // Loading skeleton
  if (isLoading) {
    return (
      <div className={`space-y-4 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded mb-3 w-1/2"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="bg-gray-100 p-3 rounded-lg">
                <div className="h-4 bg-gray-200 rounded mb-2 w-3/4"></div>
                <div className="h-6 bg-gray-200 rounded w-1/2"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }
  // No data state
  if (!metadata) {
    return (
      <div className={`space-y-4 ${className}`}>
        <h3 className="text-lg font-semibold text-gray-900">
          Analysis Metadata
        </h3>
        <div className="text-center py-8 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
          <div className="text-gray-400 text-4xl mb-2">📊</div>
          <p className="text-gray-600">No analysis data available</p>
          <p className="text-sm text-gray-500 mt-1">
            Run spatial analysis to view metadata
          </p>
        </div>
      </div>
    );
  }
  return (
    <div className={`space-y-4 ${className}`}>
      <h3 className="text-lg font-semibold text-gray-900">Analysis Metadata</h3>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
          <div className="text-sm text-blue-600 font-medium">Analysis Date</div>
          <div className="text-lg font-bold text-blue-900">
            {new Date(metadata.analysis_date).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "numeric",
            })}
          </div>
          <div className="text-xs text-blue-700 mt-1">
            {new Date(metadata.analysis_date).toLocaleTimeString("en-US", {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        </div>

        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <div className="text-sm text-green-600 font-medium">Districts</div>
          <div className="text-lg font-bold text-green-900">
            {metadata.analyzed_districts} / {metadata.total_districts}
          </div>
          <div className="text-xs text-green-700 mt-1">
            {metadata.analyzed_districts === metadata.total_districts
              ? "✅ Complete"
              : "⚠️ Partial"}
          </div>
        </div>

        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
          <div className="text-sm text-purple-600 font-medium">Data Source</div>
          <div className="text-sm font-bold text-purple-900">
            {metadata.data_source}
          </div>
          <div className="text-xs text-purple-700 mt-1">
            Backend: {metadata.processing_backend || "Flask"}
          </div>
        </div>

        <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
          <div className="text-sm text-orange-600 font-medium">Method</div>
          <div className="text-sm font-bold text-orange-900">
            {metadata.analysis_method?.replace(/_/g, " ") ||
              "Enhanced Rice Suitability"}
          </div>
          <div className="text-xs text-orange-700 mt-1">Version 2.0</div>
        </div>
      </div>

      {/* Analysis Parameters */}
      {metadata.parameters_used && (
        <div className="bg-white border rounded-lg p-4">
          <h4 className="font-semibold text-gray-900 mb-3">
            Analysis Parameters
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-1">
              <span className="text-sm text-gray-600">Time Period</span>
              <div className="font-medium">
                {metadata.parameters_used.analysis_period?.start_year} -{" "}
                {metadata.parameters_used.analysis_period?.end_year}
              </div>
            </div>

            <div className="space-y-1">
              <span className="text-sm text-gray-600">Season Filter</span>
              <div className="font-medium capitalize">
                {metadata.parameters_used.season_filter || "All Seasons"}
              </div>
            </div>

            <div className="space-y-1">
              <span className="text-sm text-gray-600">Aggregation</span>
              <div className="font-medium capitalize">
                {metadata.parameters_used.aggregation_method || "Mean"}
              </div>
            </div>

            <div className="space-y-1">
              <span className="text-sm text-gray-600">Climate Parameters</span>
              <div className="font-medium">
                {metadata.parameters_used.climate_parameters === "all"
                  ? "All Parameters"
                  : metadata.parameters_used.climate_parameters}
              </div>
            </div>
          </div>

          {/* Additional Details */}
          <div className="mt-4 pt-3 border-t border-gray-200 text-sm text-gray-600">
            <div className="flex flex-wrap gap-4">
              <span>
                📊 Output:{" "}
                {metadata.parameters_used.output_format?.toUpperCase() ||
                  "GeoJSON"}
              </span>
              <span>
                🗺️ Geometry:{" "}
                {metadata.parameters_used.include_geometry
                  ? "Included"
                  : "Excluded"}
              </span>
              <span>
                📍 Districts:{" "}
                {metadata.parameters_used.districts === "all"
                  ? "All Available"
                  : metadata.parameters_used.districts}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
