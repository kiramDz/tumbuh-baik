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
import type { TwoLevelAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

export interface FSCIFiltersProps {
  analysisParams?: TwoLevelAnalysisParams;
  level?: "kabupaten" | "kecamatan";
  onParamsChange?: (params: TwoLevelAnalysisParams) => void;
  onLevelChange?: (level: "kabupaten" | "kecamatan") => void;
  onReset?: () => void;
  onApply?: () => void;
  isLoading?: boolean;
  className?: string;
}

export function FSCIFilters({
  analysisParams,
  level = "kabupaten",
  onParamsChange,
  onLevelChange,
  onReset,
  onApply,
  isLoading = false,
  className,
}: FSCIFiltersProps) {
  const defaultParams = {
    year_start: 2018,
    year_end: 2024,
    bps_start_year: 2018,
    bps_end_year: 2024,
    season: "all" as const,
    aggregation: "mean" as const,
    districts: "all",
  };

  const safeParamsMerge = (external?: TwoLevelAnalysisParams) => ({
    year_start: external?.year_start ?? defaultParams.year_start,
    year_end: external?.year_end ?? defaultParams.year_end,
    bps_start_year: external?.bps_start_year ?? defaultParams.bps_start_year,
    bps_end_year: external?.bps_end_year ?? defaultParams.bps_end_year,
    season: external?.season ?? defaultParams.season,
    aggregation: external?.aggregation ?? defaultParams.aggregation,
    districts: external?.districts ?? defaultParams.districts,
  });

  const [filters, setFilters] = useState(() => safeParamsMerge(analysisParams));
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
  const updateFilter = (key: keyof TwoLevelAnalysisParams, value: any) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);

    // Auto-update parent component
    if (onParamsChange) {
      onParamsChange(newFilters);
    }
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

  // Handle apply (manual trigger)
  const handleApply = () => {
    if (onParamsChange) {
      onParamsChange(filters);
    }
    if (onApply) {
      onApply();
    }
    setHasChanges(false);
  };

  // Year range options
  const yearOptions = Array.from({ length: 7 }, (_, i) => 2018 + i);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center">
            <Icons.sliders className="h-5 w-5 mr-2" />
            FSCI Analysis Parameters
          </div>
          {hasChanges && (
            <Badge variant="secondary" className="text-xs">
              Unsaved Changes
            </Badge>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Analysis Level Selection */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Analysis Level</Label>
          <Select
            value={level}
            onValueChange={(value: "kabupaten" | "kecamatan") => {
              if (onLevelChange) {
                onLevelChange(value);
              }
            }}
            disabled={isLoading}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="kabupaten">
                <div className="flex items-center space-x-2">
                  <Icons.map className="h-4 w-4" />
                  <span>Kabupaten Level</span>
                </div>
              </SelectItem>
              <SelectItem value="kecamatan">
                <div className="flex items-center space-x-2">
                  <Icons.mapPin className="h-4 w-4" />
                  <span>Kecamatan Level</span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* Climate Data Parameters */}
        <div className="space-y-4">
          <Label className="text-sm font-medium flex items-center">
            <Icons.cloud className="h-4 w-4 mr-2" />
            Climate Data Range
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

        {/* BPS Production Data Parameters */}
        <div className="space-y-4">
          <Label className="text-sm font-medium flex items-center">
            <Icons.barChart className="h-4 w-4 mr-2" />
            Production Data Range
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

        {/* Season Filter */}
        <div className="space-y-2">
          <Label className="text-sm font-medium flex items-center">
            <Icons.sun className="h-4 w-4 mr-2" />
            Growing Season
          </Label>
          <Select
            value={filters.season}
            onValueChange={(value) => updateFilter("season", value)}
            disabled={isLoading}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">
                <div className="flex items-center space-x-2">
                  <Icons.calendar className="h-4 w-4" />
                  <span>All Seasons</span>
                </div>
              </SelectItem>
              <SelectItem value="wet">
                <div className="flex items-center space-x-2">
                  <Icons.cloudRain className="h-4 w-4" />
                  <span>Wet Season (Oct-Mar)</span>
                </div>
              </SelectItem>
              <SelectItem value="dry">
                <div className="flex items-center space-x-2">
                  <Icons.sun className="h-4 w-4" />
                  <span>Dry Season (Apr-Sep)</span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Aggregation Method */}
        <div className="space-y-2">
          <Label className="text-sm font-medium flex items-center">
            <Icons.calculator className="h-4 w-4" />
            Aggregation Method
          </Label>
          <Select
            value={filters.aggregation}
            onValueChange={(value) => updateFilter("aggregation", value)}
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
              <SelectItem value="percentile">
                <div className="flex items-center space-x-2">
                  <Icons.percent className="h-4 w-4" />
                  <span>Percentile</span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
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

        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="text-xs font-medium text-gray-700 mb-2">
            Current Analysis Configuration:
          </div>
          <div className="text-xs text-gray-600 space-y-1">
            <div>
              • Climate Period: {filters.year_start} - {filters.year_end}
            </div>
            <div>
              • Production Period: {filters.bps_start_year} -{" "}
              {filters.bps_end_year}
            </div>
            <div>
              • Season:{" "}
              {filters.season === "all"
                ? "All Seasons"
                : filters.season === "wet"
                ? "Wet Season"
                : "Dry Season"}
            </div>
            <div>• Aggregation: {filters.aggregation}</div>
            <div>
              • Level: {level === "kabupaten" ? "Kabupaten" : "Kecamatan"}
            </div>
            <div>
              • Scope:{" "}
              {filters.districts === "all"
                ? "All Districts"
                : filters.districts.charAt(0).toUpperCase() +
                  filters.districts.slice(1)}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
