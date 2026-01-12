"use client";

import { useState, useCallback, useEffect } from "react";
import { useTwoLevelAnalysis } from "@/hooks/use-twoLevelAnalysis";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Icons } from "@/app/dashboard/_components/icons";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface TwoLevelFiltersProps {
  className?: string;
  onApplyFilters?: () => void;
  onParametersChange?: (hasChanges: boolean) => void;
}

export function TwoLevelFilters({
  className,
  onApplyFilters,
  onParametersChange,
}: TwoLevelFiltersProps) {
  const {
    analysisParams,
    loading,
    error,
    updateParams,
    resetParams,
    fetchTwoLevelAnalysis,
    exportAnalysisToCsv,
    exporting,
  } = useTwoLevelAnalysis();

  // Local statte for form inputs
  const [localParams, setLocalParams] = useState(analysisParams);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Sync local state with global params
  useEffect(() => {
    setLocalParams(analysisParams);
    setHasUnsavedChanges(false);
  }, [analysisParams]);

  // Check for unsaved changes
  useEffect(() => {
    const hasChanges =
      JSON.stringify(localParams) !== JSON.stringify(analysisParams);
    setHasUnsavedChanges(hasChanges);
    onParametersChange?.(hasChanges);
  }, [localParams, analysisParams, onParametersChange]);

  // Apply filters
  const handleApplyFilters = useCallback(async () => {
    try {
      updateParams(localParams);
      await fetchTwoLevelAnalysis(localParams);
      onApplyFilters?.();
      setHasUnsavedChanges(false);
    } catch (error) {
      console.error("Failed to apply filters:", error);
    }
  }, [localParams, updateParams, fetchTwoLevelAnalysis, onApplyFilters]);

  // Reset filters
  const handleResetFilters = useCallback(() => {
    resetParams();
    setHasUnsavedChanges(false);
  }, [resetParams]);

  // Update individual parameters
  const updateLocalParam = useCallback(
    <K extends keyof typeof localParams>(
      key: K,
      value: (typeof localParams)[K]
    ) => {
      setLocalParams((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  // Export handler
  const handleExport = useCallback(async () => {
    try {
      await exportAnalysisToCsv();
    } catch (error) {
      console.error("Export failed:", error);
    }
  }, [exportAnalysisToCsv]);

  // Configuration options
  const currentYear = new Date().getFullYear();
  const yearOptions = Array.from({ length: 7 }, (_, i) => 2018 + i);

  const seasonOptions = [
    {
      value: "all",
      label: "Annual (Full Year)",
      description: "Complete year analysis",
      icon: Icons.calendarDays,
    },
    {
      value: "wet",
      label: "Wet Season",
      description: "November - April (Monsoon)",
      icon: Icons.cloudRain,
    },
    {
      value: "dry",
      label: "Dry Season",
      description: "May - October (Dry period)",
      icon: Icons.trendingUp,
    },
  ];

  const aggregationOptions = [
    {
      value: "mean",
      label: "Average",
      description: "Mean aggregation across time period (balanced approach)",
      technical: "μ = Σx/n",
    },
    {
      value: "median",
      label: "Median",
      description: "Median aggregation (robust to outliers)",
      technical: "Q2 (50th percentile)",
    },
    {
      value: "percentile",
      label: "Percentile (75th)",
      description: "Upper quartile aggregation (conservative estimate)",
      technical: "Q3 (75th percentile)",
    },
  ];

  // Validation helpers
  const getClimateYearSpan = () => {
    if (localParams.year_start && localParams.year_end) {
      return localParams.year_end - localParams.year_start + 1;
    }
    return 0;
  };

  const getBPSYearSpan = () => {
    if (localParams.bps_start_year && localParams.bps_end_year) {
      return localParams.bps_end_year - localParams.bps_start_year + 1;
    }
    return 0;
  };

  const isValidConfiguration = () => {
    return (
      localParams.year_start &&
      localParams.year_end &&
      localParams.bps_start_year &&
      localParams.bps_end_year &&
      localParams.year_start <= localParams.year_end &&
      localParams.bps_start_year <= localParams.bps_end_year
    );
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center justify-between text-lg">
          <div className="flex items-center">
            <Icons.filter className="h-5 w-5 mr-2" />
            Two-Level Analysis Filters
          </div>
          {hasUnsavedChanges && (
            <Badge
              variant="outline"
              className="text-amber-600 border-amber-300"
            >
              <Icons.clock className="h-3 w-3 mr-1" />
              Unsaved Changes
            </Badge>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Climate Data Period */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Icons.cloudRain className="h-4 w-4 mr-2 text-blue-600" />
              <Label className="font-medium">Climate Data Period</Label>
            </div>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Icons.info className="h-4 w-4 text-gray-400" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Climate data includes temperature, precipitation, and
                    drought indices for FSCI calculation
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="climate-start" className="text-sm">
                Start Year
              </Label>
              <Select
                value={localParams.year_start?.toString()}
                onValueChange={(value) =>
                  updateLocalParam("year_start", parseInt(value))
                }
              >
                <SelectTrigger id="climate-start">
                  <SelectValue placeholder="Select start year" />
                </SelectTrigger>
                <SelectContent>
                  {yearOptions.map((year) => (
                    <SelectItem key={year} value={year.toString()}>
                      <div className="flex items-center justify-between w-full">
                        <span>{year}</span>
                        {year === currentYear - 1 && (
                          <Badge variant="secondary" className="ml-2 text-xs">
                            Latest
                          </Badge>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="climate-end" className="text-sm">
                End Year
              </Label>
              <Select
                value={localParams.year_end?.toString()}
                onValueChange={(value) =>
                  updateLocalParam("year_end", parseInt(value))
                }
              >
                <SelectTrigger id="climate-end">
                  <SelectValue placeholder="Select end year" />
                </SelectTrigger>
                <SelectContent>
                  {yearOptions
                    .filter(
                      (year) =>
                        !localParams.year_start ||
                        year >= localParams.year_start
                    )
                    .map((year) => (
                      <SelectItem key={year} value={year.toString()}>
                        <div className="flex items-center justify-between w-full">
                          <span>{year}</span>
                          {year === currentYear - 1 && (
                            <Badge variant="secondary" className="ml-2 text-xs">
                              Latest
                            </Badge>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Climate Period Summary */}
          {localParams.year_start && localParams.year_end && (
            <div className="text-sm bg-blue-50 p-3 rounded-lg border border-blue-200">
              <div className="flex items-center justify-between mb-1">
                <strong>Climate Analysis Period:</strong>
                <Badge variant="outline" className="text-blue-700">
                  {getClimateYearSpan()} years
                </Badge>
              </div>
              <div className="text-blue-700">
                {localParams.year_start} - {localParams.year_end}
                {getClimateYearSpan() >= 5 && (
                  <span className="ml-2 text-xs">
                    • Sufficient for trend analysis
                  </span>
                )}
              </div>
            </div>
          )}
        </div>

        <Separator />

        {/* BPS Production Data Period */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Icons.database className="h-4 w-4 mr-2 text-green-600" />
              <Label className="font-medium">BPS Production Data Period</Label>
            </div>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Icons.info className="h-4 w-4 text-gray-400" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    BPS (Statistics Indonesia) rice production data in tons per
                    kabupaten
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="bps-start" className="text-sm">
                BPS Start Year
              </Label>
              <Select
                value={localParams.bps_start_year?.toString()}
                onValueChange={(value) =>
                  updateLocalParam("bps_start_year", parseInt(value))
                }
              >
                <SelectTrigger id="bps-start">
                  <SelectValue placeholder="Select BPS start year" />
                </SelectTrigger>
                <SelectContent>
                  {yearOptions.map((year) => (
                    <SelectItem key={year} value={year.toString()}>
                      {year}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="bps-end" className="text-sm">
                BPS End Year
              </Label>
              <Select
                value={localParams.bps_end_year?.toString()}
                onValueChange={(value) =>
                  updateLocalParam("bps_end_year", parseInt(value))
                }
              >
                <SelectTrigger id="bps-end">
                  <SelectValue placeholder="Select BPS end year" />
                </SelectTrigger>
                <SelectContent>
                  {yearOptions
                    .filter(
                      (year) =>
                        !localParams.bps_start_year ||
                        year >= localParams.bps_start_year
                    )
                    .map((year) => (
                      <SelectItem key={year} value={year.toString()}>
                        {year}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* BPS Period Summary */}
          {localParams.bps_start_year && localParams.bps_end_year && (
            <div className="text-sm bg-green-50 p-3 rounded-lg border border-green-200">
              <div className="flex items-center justify-between mb-1">
                <strong>Production Analysis Period:</strong>
                <Badge variant="outline" className="text-green-700">
                  {getBPSYearSpan()} years
                </Badge>
              </div>
              <div className="text-green-700">
                {localParams.bps_start_year} - {localParams.bps_end_year}
                {getBPSYearSpan() >= 3 && (
                  <span className="ml-2 text-xs">
                    • Good for correlation analysis
                  </span>
                )}
              </div>
            </div>
          )}
        </div>

        <Separator />

        {/* Analysis Configuration */}
        <div className="space-y-4">
          <div className="flex items-center">
            <Icons.barChart className="h-4 w-4 mr-2 text-purple-600" />
            <Label className="font-medium">Analysis Configuration</Label>
          </div>

          <div className="space-y-4">
            {/* Season Filter */}
            <div>
              <Label htmlFor="season" className="text-sm mb-2 block">
                Season Filter
              </Label>
              <Select
                value={localParams.season}
                onValueChange={(value) =>
                  updateLocalParam("season", value as "wet" | "dry" | "all")
                }
              >
                <SelectTrigger id="season">
                  <SelectValue placeholder="Select season" />
                </SelectTrigger>
                <SelectContent>
                  {seasonOptions.map((option) => {
                    const Icon = option.icon;
                    return (
                      <SelectItem key={option.value} value={option.value}>
                        <div className="flex items-center">
                          <Icon className="h-4 w-4 mr-2" />
                          <div>
                            <div className="font-medium">{option.label}</div>
                            <div className="text-xs text-gray-600">
                              {option.description}
                            </div>
                          </div>
                        </div>
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>

            {/* Aggregation Method */}
            <div>
              <Label htmlFor="aggregation" className="text-sm mb-2 block">
                Aggregation Method
              </Label>
              <Select
                value={localParams.aggregation}
                onValueChange={(value) =>
                  updateLocalParam(
                    "aggregation",
                    value as "mean" | "median" | "percentile"
                  )
                }
              >
                <SelectTrigger id="aggregation">
                  <SelectValue placeholder="Select aggregation method" />
                </SelectTrigger>
                <SelectContent>
                  {aggregationOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      <div>
                        <div className="font-medium">{option.label}</div>
                        <div className="text-xs text-gray-600">
                          {option.description}
                        </div>
                        <div className="text-xs text-gray-500 font-mono">
                          {option.technical}
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Districts Filter */}
            <div>
              <Label htmlFor="districts" className="text-sm mb-2 block">
                Geographic Scope
              </Label>
              <Select
                value={localParams.districts}
                onValueChange={(value) => updateLocalParam("districts", value)}
              >
                <SelectTrigger id="districts">
                  <SelectValue placeholder="Select geographic scope" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">
                    <div className="flex items-center">
                      <div>
                        <div className="font-medium">All Districts</div>
                        <div className="text-xs text-gray-600">
                          Complete Aceh province analysis
                        </div>
                      </div>
                    </div>
                  </SelectItem>
                  {/* Specific districts can be added here */}
                  <SelectItem value="Aceh Besar">Aceh Besar</SelectItem>
                  <SelectItem value="Aceh Utara">Aceh Utara</SelectItem>
                  <SelectItem value="Aceh Timur">Aceh Timur</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        <Separator />

        {/* Current Configuration Summary */}
        <div className="space-y-3">
          <Label className="text-sm font-medium">
            Current Analysis Configuration
          </Label>

          <div className="grid grid-cols-2 gap-2">
            <Badge variant="outline" className="justify-start">
              <Icons.calendarDays className="h-3 w-3 mr-1" />
              Climate: {localParams.year_start || "?"} -{" "}
              {localParams.year_end || "?"}
            </Badge>

            <Badge variant="outline" className="justify-start">
              <Icons.database className="h-3 w-3 mr-1" />
              BPS: {localParams.bps_start_year || "?"} -{" "}
              {localParams.bps_end_year || "?"}
            </Badge>

            <Badge variant="outline" className="justify-start">
              <Icons.cloudRain className="h-3 w-3 mr-1" />
              {seasonOptions.find((s) => s.value === localParams.season)
                ?.label || "Season"}
            </Badge>

            <Badge variant="outline" className="justify-start">
              <Icons.barChart className="h-3 w-3 mr-1" />
              {aggregationOptions.find(
                (a) => a.value === localParams.aggregation
              )?.label || "Aggregation"}
            </Badge>
          </div>

          {/* Configuration Status */}
          <div className="text-xs text-gray-600">
            {isValidConfiguration() ? (
              <div className="flex items-center text-green-600">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                Configuration is valid and ready for analysis
              </div>
            ) : (
              <div className="flex items-center text-amber-600">
                <div className="w-2 h-2 bg-amber-500 rounded-full mr-2"></div>
                Please complete all date ranges to proceed
              </div>
            )}
          </div>
        </div>

        <Separator />

        {/* Action Buttons */}
        <div className="space-y-3">
          <div className="flex gap-2">
            <Button
              onClick={handleApplyFilters}
              disabled={loading || !isValidConfiguration()}
              className="flex-1"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Icons.filter className="h-4 w-4 mr-2" />
                  Apply Analysis
                  {hasUnsavedChanges && " *"}
                </>
              )}
            </Button>

            <Button
              variant="outline"
              onClick={handleResetFilters}
              disabled={loading}
            >
              <Icons.rotateCcw className="h-4 w-4 mr-2" />
              Reset
            </Button>
          </div>

          <Button
            variant="secondary"
            onClick={handleExport}
            disabled={exporting || loading || !isValidConfiguration()}
            className="w-full"
          >
            {exporting ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-2"></div>
                Exporting...
              </>
            ) : (
              <>
                <Icons.download className="h-4 w-4 mr-2" />
                Export Analysis Data
              </>
            )}
          </Button>
        </div>

        {/* Analysis Warnings */}
        {getClimateYearSpan() > 6 && (
          <div className="text-sm text-amber-700 bg-amber-50 p-3 rounded-lg border border-amber-200">
            <div className="flex items-start">
              <Icons.info className="h-4 w-4 mr-2 mt-0.5 flex-shrink-0" />
              <div>
                <strong>⚠️ Extended Analysis Period:</strong> Climate analysis
                periods longer than 6 years may take significantly more time to
                process (2-3 minutes) and could impact performance.
              </div>
            </div>
          </div>
        )}

        {/* Data Alignment Information */}
        <div className="text-xs text-gray-600 bg-gray-50 p-3 rounded-lg">
          <div className="font-medium mb-2 flex items-center">
            <Icons.info className="h-3 w-3 mr-1" />
            Analysis Guidelines:
          </div>
          <ul className="list-disc list-inside space-y-1">
            <li>
              Climate and BPS periods can overlap or be different for
              correlation analysis
            </li>
            <li>
              Minimum 3 years recommended for each dataset to ensure statistical
              significance
            </li>
            <li>
              Two-level analysis correlates climate potential (FSCI) with actual
              production (BPS)
            </li>
            <li>
              Seasonal filters affect climate data aggregation but not BPS
              yearly totals
            </li>
            <li>
              Longer periods provide more robust insights but increase
              processing time
            </li>
          </ul>
        </div>

        {/* Error Display */}
        {error && (
          <div className="text-sm text-red-700 bg-red-50 p-3 rounded-lg border border-red-200">
            <div className="flex items-start">
              <Icons.info className="h-4 w-4 mr-2 mt-0.5 flex-shrink-0 text-red-600" />
              <div>
                <strong>Analysis Error:</strong> {error}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
export type { TwoLevelFiltersProps };
