"use client";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Icons } from "@/app/dashboard/_components/icons";
import { cn } from "@/lib/utils";

export interface FSILegendProps {
  /** Current FSI statistics for dynamic legend */
  stats?: {
    total_regions: number;
    avg_fsi: number;
    sangat_tinggi_count: number;
    tinggi_count: number;
    sedang_count: number;
    rendah_count: number;
  };
  /** Selected FSI range for highlighting */
  selectedRange?: "sangat_tinggi" | "tinggi" | "sedang" | "rendah" | null;
  /** Callback when legend item is clicked */
  onRangeSelect?: (
    range: "sangat_tinggi" | "tinggi" | "sedang" | "rendah" | null
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

const FSI_RANGES = [
  {
    id: "sangat_tinggi" as const,
    label: "Sangat Tinggi",
    label_en: "Very High",
    range: "80 - 100",
    color: "#22c55e", // Green-600 - Excellent FSI
    bgColor: "#D1FAE5", // Green-100
    description:
      "Ketahanan pangan sangat optimal dengan sumber daya alam berkualitas tinggi",
    description_en: "Optimal food security with high-quality natural resources",
    icon: Icons.trendingUp,
  },
  {
    id: "tinggi" as const,
    label: "Tinggi",
    label_en: "High",
    range: "60 - 79",
    color: "#84cc16", // Lime-500 - Good FSI
    bgColor: "#ECFCCB", // Lime-100
    description: "Ketahanan pangan stabil dengan ketersediaan yang baik",
    description_en: "Stable food security with good availability",
    icon: Icons.checkCircle,
  },
  {
    id: "sedang" as const,
    label: "Sedang",
    label_en: "Medium",
    range: "40 - 59",
    color: "#f59e0b", // Amber-500 - Moderate FSI
    bgColor: "#FEF3C7", // Amber-100
    description: "Ketahanan pangan moderat dengan tantangan tertentu",
    description_en: "Moderate food security with certain challenges",
    icon: Icons.alertTriangle,
  },
  {
    id: "rendah" as const,
    label: "Rendah",
    label_en: "Low",
    range: "< 40",
    color: "#ef4444", // Red-500 - Poor FSI
    bgColor: "#FEE2E2", // Red-100
    description: "Ketahanan pangan rendah memerlukan perhatian khusus",
    description_en: "Low food security requiring special attention",
    icon: Icons.alertCircle,
  },
] as const;

export function FSILegend({
  stats,
  selectedRange,
  onRangeSelect,
  compact = false,
  showStats = true,
  interactive = false,
  isLoading = false,
  className,
}: FSILegendProps) {
  const [hoveredRange, setHoveredRange] = useState<string | null>(null);

  // Handle legend item click
  const handleRangeClick = (rangeId: (typeof FSI_RANGES)[number]["id"]) => {
    if (!interactive) return;

    if (onRangeSelect) {
      // Toggle selection - if already selected, deselect
      const newSelection = selectedRange === rangeId ? null : rangeId;
      onRangeSelect(newSelection);
    }
  };

  // Get count for a specific range (✅ Updated for FSI ranges)
  const getRangeCount = (rangeId: string): number => {
    if (!stats) return 0;
    switch (rangeId) {
      case "sangat_tinggi":
        return stats.sangat_tinggi_count || 0;
      case "tinggi":
        return stats.tinggi_count || 0;
      case "sedang":
        return stats.sedang_count || 0;
      case "rendah":
        return stats.rendah_count || 0;
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
          FSI Performance {/* ✅ Updated from FSCI */}
        </div>
        <div className="space-y-1">
          {FSI_RANGES.map((range) => {
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
                  <div className="truncate font-medium">{range.label}</div>
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
            Food Security Index (FSI) Legend {/* ✅ Updated from FSCI */}
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
        <p className="text-xs text-gray-600 mt-1">
          Indeks Ketahanan Pangan berdasarkan Sumber Daya Alam (60%) dan
          Ketersediaan (40%)
        </p>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-4">
            <Icons.spinner className="h-6 w-6 animate-spin text-gray-400" />
            <span className="ml-2 text-sm text-gray-500">
              Loading FSI statistics...
            </span>
          </div>
        )}

        {/* Legend Items */}
        {!isLoading && (
          <div className="space-y-3">
            {FSI_RANGES.map((range) => {
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
                          <span className="text-xs text-gray-500 ml-2">
                            ({range.label_en})
                          </span>
                        </div>
                        <div className="text-xs text-gray-600 font-mono">
                          FSI Score: {range.range} {/* ✅ Updated from FSCI */}
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
                          kecamatan ({percentage.toFixed(1)}%)
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
                      Filter aktif
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
                Ringkasan Analisis FSI
              </div>
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <div className="text-gray-500">Total Kecamatan</div>
                  <div className="text-lg font-bold text-gray-900">
                    {stats.total_regions.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">Rata-rata FSI</div>
                  <div className="text-lg font-bold text-gray-900">
                    {stats.avg_fsi.toFixed(1)}
                  </div>
                </div>
              </div>

              {/* FSI Component Information */}
              <div className="mt-3 pt-3 border-t">
                <div className="text-xs text-gray-700 mb-2">Komponen FSI</div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-600">Sumber Daya Alam</span>
                    <span className="font-medium">60%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-600">Ketersediaan</span>
                    <span className="font-medium">40%</span>
                  </div>
                </div>
              </div>

              {/* Performance Distribution */}
              <div className="mt-3 pt-3 border-t">
                <div className="text-xs text-gray-700 mb-2">
                  Distribusi Performa
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-600">
                      Sangat Tinggi + Tinggi
                    </span>
                    <span className="font-medium">
                      {(
                        getRangePercentage("sangat_tinggi") +
                        getRangePercentage("tinggi")
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-600">Sedang + Rendah</span>
                    <span className="font-medium">
                      {(
                        getRangePercentage("sedang") +
                        getRangePercentage("rendah")
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
            Klik kategori untuk memfilter peta
          </div>
        )}

        {/* FSI Methodology Note */}
        <div className="text-xs text-gray-500 bg-blue-50 p-2 rounded border-l-2 border-blue-200">
          <div className="flex items-start">
            <Icons.info className="h-3 w-3 mr-1 mt-0.5 text-blue-500" />
            <div>
              <div className="font-medium text-blue-700">Metodologi FSI:</div>
              <div className="mt-1">
                Food Security Index menggabungkan analisis sumber daya alam
                (keberlanjutan iklim) dan ketersediaan pangan berdasarkan data
                iklim NASA POWER 2018-2024.
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
