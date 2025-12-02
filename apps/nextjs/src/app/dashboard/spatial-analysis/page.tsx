"use client";

import { useState } from "react";
import { useSpatialAnalysis } from "@/hooks/use-spatialAnalysis";
import {
  SpatialMap,
  SpatialFilters,
  MapLegend,
  MetadataPanel,
  ExportCsvButton,
} from "@/components/spatial";
import type { SpatialAnalysisParams } from "@/components/spatial";

export default function SpatialAnalysisPage() {
  const {
    analysisData,
    districts,
    parameters,
    isLoading,
    isAnalysisLoading,
    error,
    runAnalysis,
    clearError,
  } = useSpatialAnalysis();

  // Local state for UI interactions
  const [selectedFeature, setSelectedFeature] = useState<any>(null);
  const [currentParams, setCurrentParams] = useState<SpatialAnalysisParams>({
    districts: "all",
    parameters: "all",
    year_start: 2020,
    year_end: 2023,
    season: "all",
    aggregation: "mean",
  });

  // Handle filter changes and trigger new analysis
  const handleFiltersChange = async (newParams: SpatialAnalysisParams) => {
    console.log("🔄 Applying new filters:", newParams);
    setCurrentParams(newParams);
    await runAnalysis(newParams);
  };

  // Handle map feature interactions
  const handleFeatureClick = (feature: any) => {
    setSelectedFeature(feature);
    console.log("📍 Selected district:", feature.properties.NAME_3);
  };

  const handleFeatureHover = (feature: any) => {
    console.log("🖱️ Hovering over:", feature.properties.NAME_3);
  };
  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Page Header */}
      <div className="border-b bg-white px-6 py-4 shadow-sm">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              🗺️ Spatial Analysis
            </h1>
            <p className="text-sm text-gray-600 mt-1">
              Rice Suitability Analysis for Aceh Districts
            </p>
          </div>

          {/* Status Badge */}
          <div className="flex items-center space-x-4">
            {/* Data status */}
            <div className="flex items-center space-x-2">
              {error ? (
                <div className="flex items-center space-x-1">
                  <div className="h-2 w-2 bg-red-400 rounded-full"></div>
                  <span className="text-sm text-red-600">Error</span>
                  <button
                    onClick={clearError}
                    className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded hover:bg-red-200 transition-colors"
                  >
                    Clear
                  </button>
                </div>
              ) : isLoading ? (
                <div className="flex items-center space-x-1">
                  <div className="h-2 w-2 bg-yellow-400 rounded-full animate-pulse"></div>
                  <span className="text-sm text-yellow-600">Loading...</span>
                </div>
              ) : (
                <div className="flex items-center space-x-1">
                  <div className="h-2 w-2 bg-green-400 rounded-full"></div>
                  <span className="text-sm text-gray-600">
                    {analysisData?.features?.length || 0} districts analyzed
                  </span>
                </div>
              )}
            </div>

            {/* Analysis status */}
            {isAnalysisLoading && (
              <div className="flex items-center space-x-2 bg-blue-50 px-3 py-1 rounded-full">
                <div className="w-3 h-3 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                <span className="text-sm text-blue-600">
                  Running Analysis...
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-start">
              <div className="text-red-400 mr-3 mt-0.5">⚠️</div>
              <div className="flex-1">
                <h4 className="text-sm font-medium text-red-800">
                  Analysis Error
                </h4>
                <p className="text-sm text-red-600 mt-1">{error}</p>
                <div className="mt-3">
                  <button
                    onClick={() => handleFiltersChange(currentParams)}
                    className="text-sm bg-red-100 text-red-700 px-3 py-1 rounded hover:bg-red-200 transition-colors"
                  >
                    Retry Analysis
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Success notification */}
        {analysisData && !error && !isAnalysisLoading && (
          <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="flex items-center">
              <div className="text-green-400 mr-2">✅</div>
              <div>
                <span className="text-sm text-green-800 font-medium">
                  Analysis Complete:
                </span>
                <span className="text-sm text-green-700 ml-1">
                  {analysisData.features?.length} districts processed from{" "}
                  {
                    analysisData.metadata?.parameters_used?.analysis_period
                      ?.start_year
                  }
                  -
                  {
                    analysisData.metadata?.parameters_used?.analysis_period
                      ?.end_year
                  }
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Filters */}
        <div className="w-80 bg-white border-r border-gray-200 overflow-y-auto">
          <div className="p-4">
            <SpatialFilters
              districts={districts}
              parameters={parameters}
              currentParams={currentParams}
              onFiltersChange={handleFiltersChange}
              isLoading={isLoading}
              isAnalysisLoading={isAnalysisLoading}
            />
          </div>
        </div>

        {/* Center - Map Area */}
        <div className="flex-1 flex flex-col">
          {/* Map Container */}
          <div className="flex-1 relative bg-gray-100">
            <SpatialMap
              data={analysisData}
              isLoading={isLoading || isAnalysisLoading}
              onFeatureClick={handleFeatureClick}
              onFeatureHover={handleFeatureHover}
            />

            {/* Map overlay info */}
            {selectedFeature && (
              <div className="absolute bottom-4 left-4 bg-white bg-opacity-95 p-4 rounded-lg shadow-lg z-[1000] max-w-sm border">
                <h4 className="font-semibold text-gray-900 mb-2">
                  📍 {selectedFeature.properties.NAME_3}
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Regency:</span>
                    <span className="font-medium">
                      {selectedFeature.properties.NAME_2}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Score:</span>
                    <span
                      className="font-bold text-lg"
                      style={{
                        color:
                          selectedFeature.properties.suitability_score >= 70
                            ? "#16a34a"
                            : selectedFeature.properties.suitability_score >= 55
                            ? "#eab308"
                            : selectedFeature.properties.suitability_score >= 40
                            ? "#f97316"
                            : "#dc2626",
                      }}
                    >
                      {selectedFeature.properties.suitability_score?.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Classification:</span>
                    <span className="font-medium">
                      {selectedFeature.properties.classification}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Risk Level:</span>
                    <span className="capitalize font-medium">
                      {selectedFeature.properties.overall_risk}
                    </span>
                  </div>
                </div>

                <button
                  onClick={() => setSelectedFeature(null)}
                  className="mt-3 text-xs text-gray-500 hover:text-gray-700 underline"
                >
                  Close details
                </button>
              </div>
            )}
          </div>

          {/* Bottom Panel - Metadata */}
          <div className="h-48 bg-white border-t border-gray-200 overflow-y-auto">
            <MetadataPanel
              metadata={analysisData?.metadata || null}
              isLoading={isLoading || isAnalysisLoading}
              className="p-4"
            />
          </div>
        </div>

        {/* Right Sidebar - Legend & Controls */}
        <div className="w-72 bg-white border-l border-gray-200 overflow-y-auto">
          <div className="p-4 space-y-6">
            {/* Map Legend */}
            <div>
              <MapLegend showTitle />
            </div>

            {/* Export Section */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">
                Export Data
              </h3>
              <ExportCsvButton
                data={analysisData}
                isLoading={isLoading || isAnalysisLoading}
                disabled={!analysisData || !!error}
                className="mb-3"
              />

              {analysisData && (
                <div className="text-xs text-gray-500 space-y-1">
                  <p>📊 {analysisData.features?.length} districts ready</p>
                  <p>
                    📅 Period:{" "}
                    {
                      analysisData.metadata?.parameters_used?.analysis_period
                        ?.start_year
                    }
                    -
                    {
                      analysisData.metadata?.parameters_used?.analysis_period
                        ?.end_year
                    }
                  </p>
                  <p>
                    🌊 Season:{" "}
                    {analysisData.metadata?.parameters_used?.season_filter}
                  </p>
                </div>
              )}
            </div>

            {/* Quick Statistics */}
            {analysisData && (
              <div className="bg-gray-50 rounded-lg p-4 border">
                <h4 className="font-medium text-gray-900 mb-3">Quick Stats</h4>
                <div className="space-y-2 text-sm">
                  {/* Score distribution */}
                  {(() => {
                    const scores =
                      analysisData.features?.map(
                        (f) => f.properties.suitability_score
                      ) || [];
                    const excellent = scores.filter((s) => s >= 85).length;
                    const good = scores.filter((s) => s >= 70 && s < 85).length;
                    const fair = scores.filter((s) => s >= 55 && s < 70).length;
                    const marginal = scores.filter(
                      (s) => s >= 40 && s < 55
                    ).length;
                    const poor = scores.filter((s) => s < 40).length;

                    return (
                      <div className="space-y-1">
                        <div className="flex justify-between">
                          <span className="text-green-700">🟢 Excellent:</span>
                          <span className="font-medium">{excellent}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-green-600">🟢 Good:</span>
                          <span className="font-medium">{good}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-yellow-600">🟡 Fair:</span>
                          <span className="font-medium">{fair}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-orange-600">🟠 Marginal:</span>
                          <span className="font-medium">{marginal}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-red-600">🔴 Poor:</span>
                          <span className="font-medium">{poor}</span>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              </div>
            )}

            {/* Map Instructions */}
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <h4 className="font-medium text-blue-900 mb-2">Map Controls</h4>
              <ul className="text-xs text-blue-700 space-y-1">
                <li>• Hover over districts to highlight</li>
                <li>• Click districts for detailed popup</li>
                <li>• Use mouse wheel to zoom in/out</li>
                <li>• Drag to pan around the map</li>
                <li>• Selected district info shows in bottom-left</li>
              </ul>
            </div>

            {/* Analysis Info */}
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
              <h4 className="font-medium text-purple-900 mb-2">
                Analysis Info
              </h4>
              <div className="text-xs text-purple-700 space-y-1">
                <p>🌡️ Temperature: Optimal 25-30°C</p>
                <p>🌧️ Precipitation: 150-300 mm/month</p>
                <p>💧 Humidity: 70-85% relative</p>
                <p>📊 Scoring: Weighted combination</p>
                <p>🎯 Source: NASA POWER satellite data</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
