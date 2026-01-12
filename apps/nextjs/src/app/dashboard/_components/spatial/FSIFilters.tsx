"use client";
import { useState, useEffect } from "react";
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
// ✅ Updated import to use unified interface
import type { FSIAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

export interface FSIFiltersProps {
  analysisParams?: Partial<FSIAnalysisParams>;
  onParamsChange?: (params: FSIAnalysisParams) => void;
  level?: "kabupaten" | "kecamatan"; 
  onLevelChange?: (level: "kabupaten" | "kecamatan") => void; 
  onReset?: () => void;
  onApply?: () => void;
  isLoading?: boolean;
  className?: string;
}

// ✅ Updated to use complete FSIAnalysisParams
type CompleteAnalysisParams = Required<FSIAnalysisParams>;

export function FSIFilters({
  analysisParams,
  onParamsChange,
  onReset,
  onApply,
  isLoading = false,
  className,
}: FSIFiltersProps) {
  // ✅ Updated default params to include all required FSIAnalysisParams properties
  const defaultParams: CompleteAnalysisParams = {
    districts: "all",
    year_start: 2018,
    year_end: 2024,
    bps_start_year: 2018, // ✅ Added missing property
    bps_end_year: 2024, // ✅ Added missing property
    season: "all", // ✅ Added missing property
    aggregation: "mean",
    analysis_level: "both", // ✅ Added missing property
    include_bps_data: true, // ✅ Added missing property
  };

  // ✅ Updated safe merge function to handle all FSIAnalysisParams properties
  const safeParamsMerge = (
    external?: Partial<FSIAnalysisParams> // ✅ Updated parameter type
  ): CompleteAnalysisParams => ({
    districts: external?.districts ?? defaultParams.districts,
    year_start: external?.year_start ?? defaultParams.year_start,
    year_end: external?.year_end ?? defaultParams.year_end,
    bps_start_year: external?.bps_start_year ?? defaultParams.bps_start_year, // ✅ Added
    bps_end_year: external?.bps_end_year ?? defaultParams.bps_end_year, // ✅ Added
    season: external?.season ?? defaultParams.season, // ✅ Added
    aggregation: external?.aggregation ?? defaultParams.aggregation,
    analysis_level: external?.analysis_level ?? defaultParams.analysis_level, // ✅ Added
    include_bps_data:
      external?.include_bps_data ?? defaultParams.include_bps_data, // ✅ Added
  });

  const [filters, setFilters] = useState<CompleteAnalysisParams>(
    safeParamsMerge(analysisParams)
  );
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (analysisParams) {
      const safeParams = safeParamsMerge(analysisParams);
      setFilters(safeParams);
      setHasChanges(false);
    }
  }, [analysisParams]);

  // Check for changes
  useEffect(() => {
    const initialParams = safeParamsMerge(analysisParams);
    const changed = Object.keys(filters).some(
      (key) =>
        filters[key as keyof typeof filters] !==
        initialParams[key as keyof typeof initialParams]
    );
    setHasChanges(changed);
  }, [filters, analysisParams]);

  // Handle parameter updates
  const updateFilter = <K extends keyof CompleteAnalysisParams>(
    key: K,
    value: CompleteAnalysisParams[K]
  ) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
  };

  // Handle reset
  const handleReset = () => {
    const resetParams = safeParamsMerge();
    setFilters(resetParams);
    setHasChanges(false);
    if (onParamsChange) {
      onParamsChange(resetParams);
    }
    if (onReset) {
      onReset();
    }
  };

  // Handle apply
  const handleApply = () => {
    if (onParamsChange) {
      onParamsChange(filters);
    }
    if (onApply) {
      onApply();
    }
    setHasChanges(false);
  };

  // Year range options - extended range
  const yearOptions = Array.from({ length: 8 }, (_, i) => 2018 + i); // 2018-2025

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center">
            <Icons.sliders className="h-5 w-5 mr-2" />
            FSI Analysis Parameters
          </div>
          {hasChanges && (
            <Badge variant="secondary" className="text-xs">
              Unsaved Changes
            </Badge>
          )}
        </CardTitle>
        <p className="text-xs text-gray-600 mt-1">
          Configure Food Security Index spatial analysis parameters
        </p>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Climate Data Parameters */}
        <div className="space-y-4">
          <Label className="text-sm font-medium flex items-center">
            <Icons.cloud className="h-4 w-4 mr-2" />
            Climate Data Period
          </Label>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-xs text-gray-600">Start Year</Label>
              <Select
                value={filters.year_start.toString()}
                onValueChange={(value) =>
                  updateFilter("year_start", parseInt(value))
                }
                disabled={isLoading}
              >
                <SelectTrigger>
                  <SelectValue />
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

            <div className="space-y-2">
              <Label className="text-xs text-gray-600">End Year</Label>
              <Select
                value={filters.year_end.toString()}
                onValueChange={(value) =>
                  updateFilter("year_end", parseInt(value))
                }
                disabled={isLoading}
              >
                <SelectTrigger>
                  <SelectValue />
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
          </div>
        </div>

        <Separator />

        {/* ✅ NEW: BPS Data Period */}
        <div className="space-y-4">
          <Label className="text-sm font-medium flex items-center">
            <Icons.database className="h-4 w-4 mr-2" />
            Production Data Period
          </Label>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-xs text-gray-600">BPS Start Year</Label>
              <Select
                value={filters.bps_start_year.toString()}
                onValueChange={(value) =>
                  updateFilter("bps_start_year", parseInt(value))
                }
                disabled={isLoading}
              >
                <SelectTrigger>
                  <SelectValue />
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

            <div className="space-y-2">
              <Label className="text-xs text-gray-600">BPS End Year</Label>
              <Select
                value={filters.bps_end_year.toString()}
                onValueChange={(value) =>
                  updateFilter("bps_end_year", parseInt(value))
                }
                disabled={isLoading}
              >
                <SelectTrigger>
                  <SelectValue />
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
          </div>
        </div>

        <Separator />

        {/* ✅ NEW: Season Selection */}
        <div className="space-y-2">
          <Label className="text-sm font-medium flex items-center">
            <Icons.calendar className="h-4 w-4 mr-2" />
            Seasonal Analysis
          </Label>
          <Select
            value={filters.season}
            onValueChange={(value) =>
              updateFilter("season", value as CompleteAnalysisParams["season"])
            }
            disabled={isLoading}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">
                <div className="flex items-center space-x-2">
                  <Icons.globe className="h-4 w-4" />
                  <span>All Seasons</span>
                </div>
              </SelectItem>
              <SelectItem value="wet">
                <div className="flex items-center space-x-2">
                  <Icons.cloudRain className="h-4 w-4" />
                  <span>Wet Season</span>
                </div>
              </SelectItem>
              <SelectItem value="dry">
                <div className="flex items-center space-x-2">
                  <Icons.sun className="h-4 w-4" />
                  <span>Dry Season</span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* Aggregation Method - Updated to include percentile */}
        <div className="space-y-2">
          <Label className="text-sm font-medium flex items-center">
            <Icons.calculator className="h-4 w-4 mr-2" />
            Aggregation Method
          </Label>
          <Select
            value={filters.aggregation}
            onValueChange={(value) =>
              updateFilter(
                "aggregation",
                value as CompleteAnalysisParams["aggregation"]
              )
            }
            disabled={isLoading}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="mean">
                <div className="flex items-center space-x-2">
                  <Icons.trendingUp className="h-4 w-4" />
                  <span>Mean Average</span>
                </div>
              </SelectItem>
              <SelectItem value="median">
                <div className="flex items-center space-x-2">
                  <Icons.activity className="h-4 w-4" />
                  <span>Median</span>
                </div>
              </SelectItem>
              {/* ✅ Updated: percentile option instead of max/min */}
              <SelectItem value="percentile">
                <div className="flex items-center space-x-2">
                  <Icons.barChart className="h-4 w-4" />
                  <span>Percentile Analysis</span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* ✅ NEW: Analysis Level */}
        <div className="space-y-2">
          <Label className="text-sm font-medium flex items-center">
            <Icons.layers className="h-4 w-4 mr-2" />
            Analysis Level
          </Label>
          <Select
            value={filters.analysis_level}
            onValueChange={(value) =>
              updateFilter(
                "analysis_level",
                value as CompleteAnalysisParams["analysis_level"]
              )
            }
            disabled={isLoading}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="both">
                <div className="flex items-center space-x-2">
                  <Icons.grid className="h-4 w-4" />
                  <span>Both (Kecamatan + Kabupaten)</span>
                </div>
              </SelectItem>
              <SelectItem value="kecamatan">
                <div className="flex items-center space-x-2">
                  <Icons.map className="h-4 w-4" />
                  <span>Kecamatan Level</span>
                </div>
              </SelectItem>
              <SelectItem value="kabupaten">
                <div className="flex items-center space-x-2">
                  <Icons.mapPin className="h-4 w-4" />
                  <span>Kabupaten Level</span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* ✅ NEW: BPS Data Integration Toggle */}
        <div className="space-y-2">
          <Label className="text-sm font-medium flex items-center">
            <Icons.previous className="h-4 w-4 mr-2" />
            Include Production Data
          </Label>
          <Select
            value={filters.include_bps_data.toString()}
            onValueChange={(value) =>
              updateFilter("include_bps_data", value === "true")
            }
            disabled={isLoading}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="true">
                <div className="flex items-center space-x-2">
                  <Icons.check className="h-4 w-4" />
                  <span>Include BPS Production Data</span>
                </div>
              </SelectItem>
              <SelectItem value="false">
                <div className="flex items-center space-x-2">
                  <Icons.closeX className="h-4 w-4" />
                  <span>Climate Analysis Only</span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* FSI Component Information (Updated) */}
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-sm font-medium text-blue-900 mb-3">
            FSI Methodology
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-600 rounded"></div>
                <span className="text-blue-800">Sumber Daya Alam</span>
              </div>
              <span className="font-medium text-blue-900">60%</span>
            </div>
            <div className="text-xs text-blue-700 ml-5">
              Keberlanjutan iklim dan resiliensi sumber daya
            </div>

            <div className="flex items-center justify-between text-sm mt-3">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-600 rounded"></div>
                <span className="text-blue-800">Ketersediaan</span>
              </div>
              <span className="font-medium text-blue-900">40%</span>
            </div>
            <div className="text-xs text-blue-700 ml-5">
              Proksi kecukupan pasokan pangan
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3">
          <Button
            onClick={handleApply}
            disabled={!hasChanges || isLoading}
            className="flex-1"
          >
            {isLoading ? (
              <>
                <Icons.spinner className="h-4 w-4 mr-2 animate-spin" />
                Applying...
              </>
            ) : (
              <>
                <Icons.check className="h-4 w-4 mr-2" />
                Apply Changes
              </>
            )}
          </Button>

          <Button
            variant="outline"
            onClick={handleReset}
            disabled={isLoading}
            className="flex-1"
          >
            <Icons.refresh className="h-4 w-4 mr-2" />
            Reset to Default
          </Button>
        </div>

        {/* Current Configuration Summary (Updated for FSI) */}
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="text-xs font-medium text-gray-700 mb-2">
            Current FSI Analysis Configuration:
          </div>
          <div className="text-xs text-gray-600 space-y-1">
            <div>
              • Climate Period: {filters.year_start} - {filters.year_end}
            </div>
            <div>
              • Production Period: {filters.bps_start_year} -{" "}
              {filters.bps_end_year}
            </div>
            <div>• Season: {filters.season}</div>
            <div>• Aggregation: {filters.aggregation}</div>
            <div>• Analysis Level: {filters.analysis_level}</div>
            <div>
              • BPS Data: {filters.include_bps_data ? "Included" : "Excluded"}
            </div>
            <div>
              • Districts:{" "}
              {filters.districts === "all"
                ? "All Districts"
                : filters.districts}
            </div>
            <div>
              • Components: Natural Resources (60%) + Availability (40%)
            </div>
            <div>• Data Source: NASA POWER Climate Dataset</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
