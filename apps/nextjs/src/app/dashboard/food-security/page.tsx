"use client";

import { useState, Suspense } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { FSIMap } from "@/app/dashboard/_components/spatial/FSIMap";
import { FSIFilters } from "@/app/dashboard/_components/spatial/FSIFilters";
import { FSILegend } from "@/app/dashboard/_components/spatial/FSILegend";
import { RankingTable } from "@/app/dashboard/_components/FSIRankingTable";

import type { FSIAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

// ✅ FIXED: Add proper type imports for the components
interface RegionalMetadata {
  id: string;
  name: string;
  fsi_score?: number;
  production?: number;
  [key: string]: any;
}

interface FSIStats {
  total_regions: number;
  avg_fsi_score: number;
  high_security_regions: number;
  production_correlation: number;
}

interface DashboardData {
  stats: FSIStats;
  spatial_data: any[];
  ranking_data: any[];
}

function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="h-8 bg-gray-200 rounded w-1/3 animate-pulse"></div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 h-96 bg-gray-200 rounded animate-pulse"></div>
        <div className="h-96 bg-gray-200 rounded animate-pulse"></div>
      </div>
    </div>
  );
}

function SpatialLoadingSkeleton() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2">
        <div className="h-96 bg-gray-200 rounded animate-pulse"></div>
      </div>
      <div className="space-y-4">
        <div className="h-32 bg-gray-200 rounded animate-pulse"></div>
        <div className="h-64 bg-gray-200 rounded animate-pulse"></div>
      </div>
    </div>
  );
}

export default function FSIDashboard() {
  // ✅ SIMPLIFIED: Only essential states
  const [analysisParams, setAnalysisParams] = useState<
    Partial<FSIAnalysisParams>
  >({
    year_start: 2018,
    year_end: 2024,
    bps_start_year: 2018,
    bps_end_year: 2024,
    season: "all",
    aggregation: "mean",
    districts: "all",
    analysis_level: "both",
    include_bps_data: true,
  });

  const [spatialLevel, setSpatialLevel] = useState<"kabupaten" | "kecamatan">(
    "kabupaten"
  );

  // ✅ SIMPLIFIED: Update filters function
  const updateFilters = (newParams: Partial<FSIAnalysisParams>) => {
    setAnalysisParams((prev) => ({ ...prev, ...newParams }));
  };

  const handleSpatialLevelChange = (level: "kabupaten" | "kecamatan") => {
    setSpatialLevel(level);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Food Security Index (FSI) Analysis
          </h1>
          <p className="text-muted-foreground mt-2">
            Comprehensive food security analysis for Aceh Province regions
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="font-medium">
            {spatialLevel === "kabupaten"
              ? "Kabupaten Level"
              : "Kecamatan Level"}
          </Badge>
          <Badge variant="secondary">
            {analysisParams.year_start} - {analysisParams.year_end}
          </Badge>
        </div>
      </div>

      {/* ✅ SIMPLIFIED: Only 2 tabs now */}
      <Tabs defaultValue="spatial" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="spatial">Spatial Analysis</TabsTrigger>
          <TabsTrigger value="rankings">Rankings</TabsTrigger>
        </TabsList>

        {/* Spatial Analysis Tab */}
        <TabsContent value="spatial" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Analysis Parameters</CardTitle>
            </CardHeader>
            <CardContent>
              <Suspense
                fallback={
                  <div className="h-20 bg-gray-200 rounded animate-pulse" />
                }
              >
                <FSIFilters
                  analysisParams={analysisParams as FSIAnalysisParams}
                  onParamsChange={(params: FSIAnalysisParams) => {
                    setAnalysisParams(params);
                  }}
                  level={spatialLevel}
                  onLevelChange={handleSpatialLevelChange} // ✅ FIXED: Correct parameter order
                />
              </Suspense>
            </CardContent>
          </Card>

          <Suspense fallback={<SpatialLoadingSkeleton />}>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Map Section */}
              <div className="lg:col-span-2">
                <Card className="h-[600px]">
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <span>FSI Spatial Distribution</span>
                      <Badge variant="outline">
                        {spatialLevel === "kabupaten"
                          ? "5 Kabupaten"
                          : "11 Kecamatan"}
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="h-[520px]">
                    <FSIMap
                      analysisParams={analysisParams as FSIAnalysisParams}
                      level={spatialLevel}
                      className="h-full w-full"
                    />
                  </CardContent>
                </Card>
              </div>

              {/* Sidebar */}
              <div className="space-y-4">
                {/* Legend */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">FSI Legend</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <FSILegend />
                  </CardContent>
                </Card>
              </div>
            </div>
          </Suspense>
        </TabsContent>

        {/* Rankings Tab */}
        <TabsContent value="rankings" className="space-y-6">
          <Suspense fallback={<DashboardSkeleton />}>
            <RankingTable
              analysisParams={analysisParams as FSIAnalysisParams}
              level={spatialLevel}
              maxItems={50}
              showFilters={true}
              showExport={true}
            />
          </Suspense>
        </TabsContent>
      </Tabs>
    </div>
  );
}
