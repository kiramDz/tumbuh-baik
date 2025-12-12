"use client";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Icons } from "@/app/dashboard/_components/icons";
import { cn } from "@/lib/utils";

export interface FSCILegendProps {
  /** Current FSCI statistics for dynamic legend */
  stats?: {
    total_regions: number;
    avg_fsci: number;
    excellent_count: number;
    good_count: number;
    fair_count: number;
    poor_count: number;
  };
  /** Selected FSCI range for highlighting */
  selectedRange?: "excellent" | "good" | "fair" | "poor" | null;
  /** Callback when legend item is clicked */
  onRangeSelect?: (
    range: "excellent" | "good" | "fair" | "poor" | null
  ) => void;
  /** Compact mode for sidebar display */
  compact?: boolean;
  /** Show statistics */
  showStats?: boolean;
  /** Interactive legend items */
  interactive?: boolean;
  /** Loading state */
  isLoading?: boolean;
  className?: string;
}

// FSCI Classification system

const FSCI_RANGES = [
  {
    id: "excellent" as const,
    label: "Excellent Performance",
    range: "75 - 100",
    color: "#059669", // Green-600
    bgColor: "#D1FAE5", // Green-100
    description: "Optimal food security conditions",
    icon: Icons.trendingUp,
  },
  {
    id: "good" as const,
    label: "Good Performance",
    range: "60 - 74",
    color: "#3B82F6", // Blue-500
    bgColor: "#DBEAFE", // Blue-100
    description: "Stable food security with minor concerns",
    icon: Icons.checkCircle,
  },
  {
    id: "fair" as const,
    label: "Fair Performance",
    range: "45 - 59",
    color: "#F59E0B", // Amber-500
    bgColor: "#FEF3C7", // Amber-100
    description: "Moderate food security challenges",
    icon: Icons.alertTriangle,
  },
  {
    id: "poor" as const,
    label: "Poor Performance",
    range: "< 45",
    color: "#DC2626", // Red-600
    bgColor: "#FEE2E2", // Red-100
    description: "Significant food security concerns",
    icon: Icons.alertCircle,
  },
] as const;

export function FSCILegend({
  stats,
  selectedRange,
  onRangeSelect,
  compact = false,
  showStats = true,
  interactive = false,
  isLoading = false,
  className,
}: FSCILegendProps) {
  const [hoveredRange, setHoveredRange] = useState<string | null>(null);

  // Handle legend item click
  const handleRangeClick = (rangeId: (typeof FSCI_RANGES)[number]["id"]) => {
    if (!interactive) return;

    if (onRangeSelect) {
      // Toggle selection - if already selected, deselect
      const newSelection = selectedRange === rangeId ? null : rangeId;
      onRangeSelect(newSelection);
    }
  };

  // Get count for a specific range
  const getRangeCount = (rangeId: string): number => {
    if (!stats) return 0;
    switch (rangeId) {
      case "excellent":
        return stats.excellent_count || 0;
      case "good":
        return stats.good_count || 0;
      case "fair":
        return stats.fair_count || 0;
      case "poor":
        return stats.poor_count || 0;
      default:
        return 0;
    }
  };

  // Get percentage for a range
  const getRangePercentage = (rangeId: string): number => {
    if (!stats || stats.total_regions === 0) return 0;
    return (getRangeCount(rangeId) / stats.total_regions) * 100;
  };

  if (compact) {
    return (
      <div className={cn("space-y-2", className)}>
        <div className="text-xs font-medium text-gray-700 mb-2">
          FSCI Performance
        </div>
        <div className="space-y-1">
          {FSCI_RANGES.map((range) => {
            const isSelected = selectedRange === range.id;
            const isHovered = hoveredRange === range.id;
            const count = getRangeCount(range.id);

            return (
              <div
                key={range.id}
                className={cn(
                  "flex items-center space-x-2 p-2 rounded text-xs cursor-pointer transition-all duration-200",
                  interactive && "hover:bg-gray-50",
                  isSelected && "ring-2 ring-offset-1",
                  isHovered && "bg-gray-50"
                )}
                style={
                  {
                    "--tw-ring-color": isSelected ? range.color : undefined,
                  } as React.CSSProperties
                }
                onClick={() => handleRangeClick(range.id)}
                onMouseEnter={() => interactive && setHoveredRange(range.id)}
                onMouseLeave={() => interactive && setHoveredRange(null)}
              >
                <div
                  className="w-3 h-3 rounded"
                  style={{ backgroundColor: range.color }}
                />
                <div className="flex-1 min-w-0">
                  <div className="truncate font-medium">
                    {range.label.split(" ")[0]}
                  </div>
                  <div className="text-gray-500">{range.range}</div>
                </div>
                {showStats && stats && (
                  <div className="text-right">
                    <div className="font-medium">{count}</div>
                    <div className="text-gray-500">
                      {getRangePercentage(range.id).toFixed(0)}%
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // Full legend rendering
  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center justify-between">
          <div className="flex items-center">
            <Icons.theme className="h-5 w-5 mr-2" />
            FSCI Performance Legend
          </div>
          {interactive && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onRangeSelect?.(null)}
              disabled={!selectedRange}
            >
              <Icons.closeX className="h-4 w-4" />
            </Button>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-4">
            <Icons.spinner className="h-6 w-6 animate-spin text-gray-400" />
            <span className="ml-2 text-sm text-gray-500">
              Loading statistics...
            </span>
          </div>
        )}

        {/* Legend Items */}
        {!isLoading && (
          <div className="space-y-3">
            {FSCI_RANGES.map((range) => {
              const Icon = range.icon;
              const isSelected = selectedRange === range.id;
              const isHovered = hoveredRange === range.id;
              const count = getRangeCount(range.id);
              const percentage = getRangePercentage(range.id);

              return (
                <div
                  key={range.id}
                  className={cn(
                    "group p-3 rounded-lg border transition-all duration-200 cursor-pointer",
                    interactive && "hover:shadow-md hover:scale-[1.02]",
                    isSelected && "ring-2 ring-offset-2 shadow-md",
                    !interactive && "cursor-default"
                  )}
                  style={
                    {
                      backgroundColor:
                        isHovered || isSelected ? range.bgColor : undefined,
                      borderColor: isSelected ? range.color : undefined,
                      "--tw-ring-color": isSelected ? range.color : undefined,
                    } as React.CSSProperties
                  }
                  onClick={() => handleRangeClick(range.id)}
                  onMouseEnter={() => interactive && setHoveredRange(range.id)}
                  onMouseLeave={() => interactive && setHoveredRange(null)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {/* Color Indicator */}
                      <div className="flex flex-col items-center space-y-1">
                        <div
                          className="w-4 h-4 rounded shadow-sm"
                          style={{ backgroundColor: range.color }}
                        />
                        <Icon
                          className="h-3 w-3"
                          style={{ color: range.color }}
                        />
                      </div>

                      {/* Range Info */}
                      <div>
                        <div className="font-semibold text-sm text-gray-900">
                          {range.label}
                        </div>
                        <div className="text-xs text-gray-600 font-mono">
                          FSCI Score: {range.range}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                          {range.description}
                        </div>
                      </div>
                    </div>

                    {/* Statistics */}
                    {showStats && stats && (
                      <div className="text-right">
                        <div className="text-lg font-bold text-gray-900">
                          {count}
                        </div>
                        <div className="text-xs text-gray-500">
                          regions ({percentage.toFixed(1)}%)
                        </div>

                        {/* Progress Bar */}
                        <div className="w-16 h-1 bg-gray-200 rounded-full mt-2">
                          <div
                            className="h-1 rounded-full transition-all duration-300"
                            style={{
                              backgroundColor: range.color,
                              width: `${percentage}%`,
                            }}
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Selection Indicator */}
                  {isSelected && interactive && (
                    <div
                      className="mt-2 flex items-center text-xs"
                      style={{ color: range.color }}
                    >
                      <Icons.check className="h-3 w-3 mr-1" />
                      Filter active
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Overall Statistics */}
        {!isLoading && showStats && stats && (
          <>
            <Separator />
            <div className="bg-gray-50 p-3 rounded-lg">
              <div className="text-xs font-medium text-gray-700 mb-2">
                Analysis Summary
              </div>
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <div className="text-gray-500">Total Regions</div>
                  <div className="text-lg font-bold text-gray-900">
                    {stats.total_regions.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">Average FSCI</div>
                  <div className="text-lg font-bold text-gray-900">
                    {stats.avg_fsci.toFixed(1)}
                  </div>
                </div>
              </div>

              {/* Performance Distribution */}
              <div className="mt-3 pt-3 border-t">
                <div className="text-xs text-gray-700 mb-2">
                  Performance Distribution
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-600">Excellent + Good</span>
                    <span className="font-medium">
                      {(
                        getRangePercentage("excellent") +
                        getRangePercentage("good")
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-600">Fair + Poor</span>
                    <span className="font-medium">
                      {(
                        getRangePercentage("fair") + getRangePercentage("poor")
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Interactive Help */}
        {interactive && (
          <div className="text-xs text-gray-500 flex items-center">
            <Icons.info className="h-3 w-3 mr-1" />
            Click ranges to filter map regions
          </div>
        )}
      </CardContent>
    </Card>
  );
}
