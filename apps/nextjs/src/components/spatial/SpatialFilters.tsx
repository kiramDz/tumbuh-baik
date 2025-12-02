"use client";

import { useState, useEffect } from "react";
import { Icons } from "@/app/dashboard/_components/icons";
import type {
  SpatialAnalysisParams,
  SpatialAnalysisResponse,
} from "@/lib/fetch/files.fetch";

interface District {
  id: string;
  name: string;
  kabupaten: string;
  province: string;
}

interface SpatialFiltersProps {
  districts: District[] | null;
  parameters: any | null;
  currentParams: SpatialAnalysisParams;
  onFiltersChange: (filters: SpatialAnalysisParams) => void;
  isLoading?: boolean;
  isAnalysisLoading?: boolean;
  className?: string;
}

export function SpatialFilters({
  districts,
  parameters,
  currentParams,
  onFiltersChange,
  isLoading = false,
  isAnalysisLoading = false,
  className = "",
}: SpatialFiltersProps) {
  // Form state
  const [formData, setFormData] =
    useState<SpatialAnalysisParams>(currentParams);
  const [hasChanges, setHasChanges] = useState(false);

  // Update form when currentParams change
  useEffect(() => {
    setFormData(currentParams);
    setHasChanges(false);
  }, [currentParams]);

  // Check for changes
  useEffect(() => {
    const hasChanged =
      JSON.stringify(formData) !== JSON.stringify(currentParams);
    setHasChanges(hasChanged);
  }, [formData, currentParams]);

  // Handle input changes
  const handleInputChange = (
    field: keyof SpatialAnalysisParams,
    value: any
  ) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  // Handle form submission
  const handleApplyFilters = () => {
    if (!hasChanges) return;
    onFiltersChange(formData);
  };

  const handleReset = () => {
    const defaultParams: SpatialAnalysisParams = {
      districts: "all",
      parameters: "all",
      year_start: 2020,
      year_end: 2023,
      season: "all",
      aggregation: "mean",
    };
    setFormData(defaultParams);
    onFiltersChange(defaultParams);
  };

  // District options
  const districtOptions = [
    { value: "all", label: "All Districts" },
    ...(districts?.map((d) => ({
      value: d.id || d.name,
      label: `${d.name} (${d.kabupaten})`,
    })) || []),
  ];

  // Parameter options
  const parameterOptions = [
    { value: "all", label: "All Parameters" },
    { value: "temperature", label: "Temperature Only" },
    { value: "precipitation", label: "Precipitation Only" },
    { value: "humidity", label: "Humidity Only" },
    { value: "temperature,precipitation", label: "Temp + Precipitation" },
    { value: "temperature,humidity", label: "Temp + Humidity" },
    { value: "precipitation,humidity", label: "Precipitation + Humidity" },
  ];

  // Season options
  const seasonOptions = [
    { value: "all", label: "All Seasons" },
    { value: "wet", label: "Wet Season (Nov-Apr)" },
    { value: "dry", label: "Dry Season (May-Oct)" },
  ];

  // Aggregation options
  const aggregationOptions = [
    { value: "mean", label: "Mean Average" },
    { value: "median", label: "Median Average" },
    { value: "percentile", label: "Percentile Based" },
  ];
  return (
    <div className={`space-y-6 ${className}`}>
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900">
          Analysis Filters
        </h2>

        {/* Status indicator */}
        <div className="flex items-center space-x-2">
          {isAnalysisLoading ? (
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-xs text-blue-600">Analyzing...</span>
            </div>
          ) : hasChanges ? (
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
              <span className="text-xs text-orange-600">Changes pending</span>
            </div>
          ) : (
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span className="text-xs text-green-600">Applied</span>
            </div>
          )}
        </div>
      </div>

      {/* District Selection */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          <Icons.mapPin className="inline w-4 h-4 mr-1" />
          Districts
        </label>
        <select
          value={formData.districts || "all"}
          onChange={(e) => handleInputChange("districts", e.target.value)}
          disabled={isLoading || !districts}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
        >
          {districtOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-500">
          {districts
            ? `${districts.length} districts available`
            : "Loading districts..."}
        </p>
      </div>

      {/* Climate Parameters */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          <Icons.thermometer className="inline w-4 h-4 mr-1" />
          Climate Parameters
        </label>
        <select
          value={formData.parameters || "all"}
          onChange={(e) => handleInputChange("parameters", e.target.value)}
          disabled={isLoading}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
        >
          {parameterOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-500">
          Affects suitability score calculation
        </p>
      </div>

      {/* Year Range */}
      <div className="space-y-3">
        <label className="block text-sm font-medium text-gray-700">
          <Icons.calendar className="inline w-4 h-4 mr-1" />
          Analysis Period
        </label>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-600 mb-1">
              Start Year
            </label>
            <select
              value={formData.year_start || 2020}
              onChange={(e) =>
                handleInputChange("year_start", parseInt(e.target.value))
              }
              disabled={isLoading}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {[2020, 2021, 2022, 2023].map((year) => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs text-gray-600 mb-1">End Year</label>
            <select
              value={formData.year_end || 2023}
              onChange={(e) =>
                handleInputChange("year_end", parseInt(e.target.value))
              }
              disabled={isLoading}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {[2020, 2021, 2022, 2023].map((year) => (
                <option
                  key={year}
                  value={year}
                  disabled={year < (formData.year_start || 2020)}
                >
                  {year}
                </option>
              ))}
            </select>
          </div>
        </div>

        <p className="text-xs text-gray-500">
          Period span:{" "}
          {(formData.year_end || 2023) - (formData.year_start || 2020) + 1}{" "}
          years
        </p>
      </div>

      {/* Season Filter */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          <Icons.cloud className="inline w-4 h-4 mr-1" />
          Season Filter
        </label>
        <select
          value={formData.season || "all"}
          onChange={(e) =>
            handleInputChange("season", e.target.value as "wet" | "dry" | "all")
          }
          disabled={isLoading}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          {seasonOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-500">
          Seasonal analysis based on typical rice growing patterns
        </p>
      </div>

      {/* Aggregation Method */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          <Icons.barChart className="inline w-4 h-4 mr-1" />
          Aggregation Method
        </label>
        <select
          value={formData.aggregation || "mean"}
          onChange={(e) =>
            handleInputChange(
              "aggregation",
              e.target.value as "mean" | "median" | "percentile"
            )
          }
          disabled={isLoading}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          {aggregationOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-500">
          Statistical method for combining multi-year data
        </p>
      </div>

      {/* Action Buttons */}
      <div className="space-y-3 pt-4 border-t border-gray-200">
        <button
          onClick={handleApplyFilters}
          disabled={!hasChanges || isAnalysisLoading || isLoading}
          className={`
            w-full py-3 px-4 rounded-lg font-medium transition-all
            ${
              hasChanges && !isAnalysisLoading
                ? "bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg"
                : "bg-gray-200 text-gray-500 cursor-not-allowed"
            }
          `}
        >
          {isAnalysisLoading ? (
            <div className="flex items-center justify-center space-x-2">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Running Analysis...</span>
            </div>
          ) : hasChanges ? (
            <div className="flex items-center justify-center space-x-2">
              <Icons.play className="w-4 h-4" />
              <span>Apply Changes</span>
            </div>
          ) : (
            <div className="flex items-center justify-center space-x-2">
              <Icons.check className="w-4 h-4" />
              <span>Filters Applied</span>
            </div>
          )}
        </button>

        <button
          onClick={handleReset}
          disabled={isAnalysisLoading || isLoading}
          className="w-full py-2 px-4 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="flex items-center justify-center space-x-1">
            <Icons.rotateCcw className="w-4 h-4" />
            <span>Reset to Defaults</span>
          </div>
        </button>
      </div>

      {/* Quick Stats */}
      {parameters && (
        <div className="bg-gray-50 rounded-lg p-3 border">
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            Data Summary
          </h4>
          <div className="space-y-1 text-xs text-gray-600">
            <p>• {districts?.length || 0} districts available</p>
            <p>• 2020-2023 climate data period</p>
            <p>• NASA POWER satellite data source</p>
            <p>• Real-time Flask analysis backend</p>
          </div>
        </div>
      )}
    </div>
  );
}
