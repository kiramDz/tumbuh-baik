"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { useTwoLevelAnalysis } from "@/hooks/use-twoLevelAnalysis";
import { useFoodSecurityAnalysis } from "@/hooks/use-foodSecurityAnalysis"; // ✅ Added for geospatial data
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Icons } from "@/app/dashboard/_components/icons";

interface TwoLevelMapProps {
  className?: string;
  onKabupatenSelect?: (kabupaten: string) => void;
  onKecamatanSelect?: (kecamatan: string) => void;
}

// ✅ Type definitions for coordinate handling
type Coordinate = [number, number];
type Polygon = Coordinate[];
type MultiPolygonCoordinates = number[][][][];

export function TwoLevelMap({
  className,
  onKabupatenSelect,
  onKecamatanSelect,
}: TwoLevelMapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);

  // ✅ Use food security hook for geospatial data + two-level hook for statistics
  const {
    data: spatialData,
    loading: spatialLoading,
    error: spatialError,
    fetchAnalysis: fetchSpatialData,
  } = useFoodSecurityAnalysis();

  const {
    analysisData: statsData,
    loading: statsLoading,
    error: statsError,
    selectedKabupaten,
    selectedKecamatan,
    selectKabupaten,
    selectKecamatan,
    fetchTwoLevelAnalysis,
  } = useTwoLevelAnalysis();

  // Layer management
  const [activeLayers, setActiveLayers] = useState({
    kecamatan: true,
    kabupaten: true,
    production: false,
    fsci: true,
  });

  const [layerGroups, setLayerGroups] = useState<{
    kecamatan: L.LayerGroup | null;
    kabupaten: L.LayerGroup | null;
  }>({ kecamatan: null, kabupaten: null });

  // ✅ Combined loading and error states
  const loading = spatialLoading || statsLoading;
  const error = spatialError || statsError;

  // Initialize map
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    // Create map
    const map = L.map(mapContainerRef.current, {
      center: [4.695135, 96.7493993], // Aceh center
      zoom: 8,
      zoomControl: false, // Custom zoom controls
    });

    // Add tile layer
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap contributors",
      maxZoom: 18,
    }).addTo(map);

    // Initialize layer groups
    const kecamatanGroup = L.layerGroup().addTo(map);
    const kabupatenGroup = L.layerGroup().addTo(map);

    setLayerGroups({
      kecamatan: kecamatanGroup,
      kabupaten: kabupatenGroup,
    });

    mapRef.current = map;

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  // Color schemes for different metrics
  const getFSCIColor = useCallback((fsciScore: number): string => {
    if (fsciScore >= 80) return "#10B981"; // Green - Lumbung Pangan Primer
    if (fsciScore >= 60) return "#F59E0B"; // Orange - Lumbung Pangan Sekunder
    if (fsciScore >= 40) return "#EF4444"; // Red - Lumbung Pangan Tersier
    return "#6B7280"; // Gray - No data
  }, []);

  const getProductionColor = useCallback((productionTons: number): string => {
    if (productionTons >= 300000) return "#1E3A8A"; // Dark blue - Very high
    if (productionTons >= 200000) return "#3B82F6"; // Blue - High
    if (productionTons >= 100000) return "#60A5FA"; // Light blue - Medium
    if (productionTons >= 50000) return "#93C5FD"; // Very light blue - Low
    return "#E5E7EB"; // Gray - Very low/no data
  }, []);

  const getPerformanceColor = useCallback((category: string): string => {
    switch (category) {
      case "overperforming":
        return "#10B981"; // Green
      case "aligned":
        return "#6366F1"; // Blue
      case "underperforming":
        return "#F59E0B"; // Orange
      default:
        return "#6B7280"; // Gray
    }
  }, []);

  // ✅ Helper to get kabupaten stats from two-level data
  const getKabupatenStats = useCallback(
    (kabupatenName: string) => {
      if (!statsData?.level_2_kabupaten_analysis?.data) return null;

      return statsData.level_2_kabupaten_analysis.data.find(
        (kabupaten) => kabupaten.kabupaten_name === kabupatenName
      );
    },
    [statsData]
  );

  // ✅ Fixed: Render kecamatan layer using spatial data
  const renderKecamatanLayer = useCallback(() => {
    if (
      !mapRef.current ||
      !layerGroups.kecamatan ||
      !spatialData?.features // ✅ Fixed: Use spatialData.features
    )
      return;

    // Clear existing kecamatan features
    layerGroups.kecamatan.clearLayers();

    spatialData.features.forEach((feature) => {
      // ✅ Fixed: Added type safety
      if (feature.geometry?.type === "MultiPolygon") {
        const coordinates = feature.geometry
          .coordinates as unknown as MultiPolygonCoordinates;

        // ✅ Fixed: Added proper typing for coordinate conversion
        const polygons: Polygon[] = coordinates.map((polygon: number[][][]) =>
          polygon[0].map(
            (coord: number[]) => [coord[1], coord[0]] as Coordinate
          )
        );

        polygons.forEach((polygon: Polygon) => {
          const fsciScore = feature.properties.fsci_score || 0;
          const kecamatanName =
            feature.properties.kecamatan_name || feature.properties.NAME_3;

          // ✅ Get production data from two-level stats if available
          const kabupatenStats = getKabupatenStats(
            feature.properties.kabupaten_name
          );
          const productionTons =
            kabupatenStats?.latest_production_tons ||
            feature.properties.latest_production_tons ||
            0;

          const color = activeLayers.fsci
            ? getFSCIColor(fsciScore)
            : activeLayers.production
            ? getProductionColor(productionTons)
            : "#3B82F6";

          const leafletPolygon = L.polygon(polygon, {
            color: color,
            fillColor: color,
            fillOpacity: selectedKecamatan === kecamatanName ? 0.8 : 0.6,
            weight: selectedKecamatan === kecamatanName ? 3 : 1,
            opacity: 1,
          });

          // Enhanced popup with two-level data
          const kabupatenInfo = kabupatenStats
            ? `
            <div class="border-t mt-2 pt-2 text-xs text-gray-600">
              <strong>Kabupaten Stats:</strong><br>
              Performance: ${kabupatenStats.performance_gap_category}<br>
              Correlation: ${kabupatenStats.climate_production_correlation.toFixed(
                3
              )}<br>
              Efficiency: ${(
                kabupatenStats.production_efficiency_score * 100
              ).toFixed(1)}%
            </div>
          `
            : "";

          leafletPolygon.bindPopup(`
            <div class="p-2 min-w-[200px]">
              <h3 class="font-semibold text-lg mb-2">${kecamatanName}</h3>
              <div class="space-y-1 text-sm">
                <div><strong>Kabupaten:</strong> ${
                  feature.properties.kabupaten_name || "N/A"
                }</div>
                <div><strong>FSCI Score:</strong> ${fsciScore.toFixed(1)}</div>
                <div><strong>FSCI Class:</strong> ${
                  feature.properties.fsci_class || "N/A"
                }</div>
                <div><strong>PCI:</strong> ${(
                  feature.properties.pci_score || 0
                ).toFixed(1)}</div>
                <div><strong>PSI:</strong> ${(
                  feature.properties.psi_score || 0
                ).toFixed(1)}</div>
                <div><strong>CRS:</strong> ${(
                  feature.properties.crs_score || 0
                ).toFixed(1)}</div>
                <div><strong>Production:</strong> ${(
                  productionTons / 1000
                ).toFixed(1)}K tons</div>
              </div>
              ${kabupatenInfo}
            </div>
          `);

          // Click handler
          leafletPolygon.on("click", () => {
            selectKecamatan(kecamatanName);
            onKecamatanSelect?.(kecamatanName);
          });

          // ✅ Fixed: Null safety check
          if (layerGroups.kecamatan) {
            leafletPolygon.addTo(layerGroups.kecamatan);
          }
        });
      }
    });
  }, [
    spatialData,
    layerGroups.kecamatan,
    activeLayers.fsci,
    activeLayers.production,
    selectedKecamatan,
    getFSCIColor,
    getProductionColor,
    selectKecamatan,
    onKecamatanSelect,
    getKabupatenStats,
  ]);

  // ✅ Fixed: Render kabupaten layer using both spatial and stats data
  const renderKabupatenLayer = useCallback(() => {
    if (
      !mapRef.current ||
      !layerGroups.kabupaten ||
      !statsData?.level_2_kabupaten_analysis?.data || // ✅ Fixed: Use correct path
      !spatialData?.features
    )
      return;

    // Clear existing kabupaten features
    layerGroups.kabupaten.clearLayers();

    // Create kabupaten boundaries and labels
    statsData.level_2_kabupaten_analysis.data.forEach((kabupaten) => {
      // Find corresponding geospatial features for this kabupaten
      const kecamatanFeatures = spatialData.features.filter(
        (feature) =>
          feature.properties.kabupaten_name === kabupaten.kabupaten_name
      );

      if (kecamatanFeatures.length === 0) return;

      // Calculate kabupaten boundary (union of kecamatan boundaries)
      let allCoordinates: Polygon[] = [];
      kecamatanFeatures.forEach((feature) => {
        // ✅ Fixed: Added type safety
        if (feature.geometry?.type === "MultiPolygon") {
          const coordinates = feature.geometry
            .coordinates as unknown as MultiPolygonCoordinates;
          coordinates.forEach((polygon: number[][][]) => {
            allCoordinates.push(
              polygon[0].map(
                (coord: number[]) => [coord[1], coord[0]] as Coordinate
              )
            );
          });
        }
      });

      if (allCoordinates.length === 0) return;

      // Create kabupaten boundary
      allCoordinates.forEach((polygon: Polygon) => {
        // ✅ Fixed: Added type safety
        const performanceCategory =
          kabupaten.performance_gap_category || "aligned";
        const color = getPerformanceColor(performanceCategory);

        const leafletPolygon = L.polygon(polygon, {
          color: color,
          fillColor: "transparent",
          weight: selectedKabupaten === kabupaten.kabupaten_name ? 4 : 2,
          opacity: 1,
          dashArray: "5, 10", // Dashed line for kabupaten boundaries
        });

        // Kabupaten popup
        leafletPolygon.bindPopup(`
          <div class="p-3 min-w-[250px]">
            <h3 class="font-bold text-lg mb-3">${kabupaten.kabupaten_name}</h3>
            <div class="grid grid-cols-2 gap-2 text-sm">
              <div><strong>FSCI Score:</strong> ${kabupaten.aggregated_fsci_score.toFixed(
                1
              )}</div>
              <div><strong>Production:</strong> ${(
                kabupaten.latest_production_tons / 1000
              ).toFixed(1)}K tons</div>
              <div><strong>Efficiency:</strong> ${(
                kabupaten.production_efficiency_score * 100
              ).toFixed(1)}%</div>
              <div><strong>Correlation:</strong> ${kabupaten.climate_production_correlation.toFixed(
                3
              )}</div>
              <div class="col-span-2"><strong>Performance:</strong> 
                <span class="ml-1 px-2 py-1 rounded text-xs font-medium" style="background-color: ${color}20; color: ${color}">
                  ${performanceCategory}
                </span>
              </div>
              <div class="col-span-2"><strong>Climate Rank:</strong> ${
                kabupaten.climate_potential_rank
              }</div>
              <div class="col-span-2"><strong>Production Rank:</strong> ${
                kabupaten.actual_production_rank || "N/A"
              }</div>
            </div>
          </div>
        `);

        // Click handler
        leafletPolygon.on("click", () => {
          selectKabupaten(kabupaten.kabupaten_name);
          onKabupatenSelect?.(kabupaten.kabupaten_name);
        });

        // ✅ Fixed: Null safety check
        if (layerGroups.kabupaten) {
          leafletPolygon.addTo(layerGroups.kabupaten);
        }
      });

      // Add kabupaten label at centroid
      if (allCoordinates.length > 0) {
        // Calculate centroid
        let totalLat = 0,
          totalLng = 0,
          pointCount = 0;
        allCoordinates.forEach((coordinates: Polygon) => {
          // ✅ Fixed: Added type safety
          coordinates.forEach((coord: Coordinate) => {
            totalLat += coord[0]; // ✅ Fixed: coord is already [lat, lng]
            totalLng += coord[1];
            pointCount++;
          });
        });

        if (pointCount > 0) {
          const centroidLat = totalLat / pointCount;
          const centroidLng = totalLng / pointCount;

          // Kabupaten label
          const label = L.divIcon({
            className: "kabupaten-label",
            html: `
              <div class="bg-white bg-opacity-90 px-2 py-1 rounded border shadow text-xs font-semibold text-gray-800 whitespace-nowrap">
                ${kabupaten.kabupaten_name}
                <div class="text-xs text-gray-600">FSCI: ${kabupaten.aggregated_fsci_score.toFixed(
                  1
                )}</div>
              </div>
            `,
            iconSize: [100, 30],
            iconAnchor: [50, 15],
          });

          // ✅ Fixed: Null safety check
          if (layerGroups.kabupaten) {
            L.marker([centroidLat, centroidLng], { icon: label }).addTo(
              layerGroups.kabupaten
            );
          }
        }
      }
    });
  }, [
    statsData,
    spatialData,
    layerGroups.kabupaten,
    selectedKabupaten,
    getPerformanceColor,
    selectKabupaten,
    onKabupatenSelect,
  ]);

  // Update layers when data or settings change
  useEffect(() => {
    if (activeLayers.kecamatan) {
      renderKecamatanLayer();
    } else {
      layerGroups.kecamatan?.clearLayers();
    }
  }, [activeLayers.kecamatan, renderKecamatanLayer]);

  useEffect(() => {
    if (activeLayers.kabupaten) {
      renderKabupatenLayer();
    } else {
      layerGroups.kabupaten?.clearLayers();
    }
  }, [activeLayers.kabupaten, renderKabupatenLayer]);

  // Map controls
  const zoomIn = useCallback(() => {
    mapRef.current?.zoomIn();
  }, []);

  const zoomOut = useCallback(() => {
    mapRef.current?.zoomOut();
  }, []);

  const resetView = useCallback(() => {
    mapRef.current?.setView([4.695135, 96.7493993], 8);
  }, []);

  // Toggle layer visibility
  const toggleLayer = useCallback((layerName: keyof typeof activeLayers) => {
    setActiveLayers((prev) => ({ ...prev, [layerName]: !prev[layerName] }));
  }, []);

  // ✅ Load both spatial and statistical data on mount
  useEffect(() => {
    fetchSpatialData(); // Load geospatial data
    fetchTwoLevelAnalysis(); // Load statistical data
  }, [fetchSpatialData, fetchTwoLevelAnalysis]);

  if (error) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <div className="text-center text-red-600">
            <p>Error loading map data: {error}</p>
            <Button
              onClick={() => {
                fetchSpatialData();
                fetchTwoLevelAnalysis();
              }}
              className="mt-2"
            >
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardContent className="p-0 relative">
        {/* Map Container */}
        <div ref={mapContainerRef} className="w-full h-[600px] rounded-lg" />

        {/* Loading Overlay */}
        {loading && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1000] rounded-lg">
            <div className="bg-white p-4 rounded-lg shadow-lg">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span>Loading two-level analysis...</span>
              </div>
            </div>
          </div>
        )}

        {/* Map Controls */}
        <div className="absolute top-4 right-4 z-[1000] space-y-2">
          {/* Zoom Controls */}
          <div className="bg-white rounded-lg shadow-lg p-1">
            <Button variant="ghost" size="sm" onClick={zoomIn}>
              <Icons.zoomIn className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={zoomOut}>
              <Icons.zoomOut className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={resetView}>
              <Icons.rotateCcw className="h-4 w-4" />
            </Button>
          </div>

          {/* Layer Controls */}
          <div className="bg-white rounded-lg shadow-lg p-3 min-w-[200px]">
            <div className="flex items-center mb-2">
              <Icons.layers className="h-4 w-4 mr-2" />
              <span className="font-medium text-sm">Map Layers</span>
            </div>

            <div className="space-y-2 text-sm">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={activeLayers.kecamatan}
                  onChange={() => toggleLayer("kecamatan")}
                  className="mr-2"
                />
                Kecamatan (Level 1)
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={activeLayers.kabupaten}
                  onChange={() => toggleLayer("kabupaten")}
                  className="mr-2"
                />
                Kabupaten (Level 2)
              </label>

              <div className="border-t pt-2 mt-2">
                <div className="text-xs text-gray-600 mb-1">Color Scheme:</div>

                <label className="flex items-center">
                  <input
                    type="radio"
                    name="colorScheme"
                    checked={activeLayers.fsci}
                    onChange={() =>
                      setActiveLayers((prev) => ({
                        ...prev,
                        fsci: true,
                        production: false,
                      }))
                    }
                    className="mr-2"
                  />
                  FSCI Score
                </label>

                <label className="flex items-center">
                  <input
                    type="radio"
                    name="colorScheme"
                    checked={activeLayers.production}
                    onChange={() =>
                      setActiveLayers((prev) => ({
                        ...prev,
                        fsci: false,
                        production: true,
                      }))
                    }
                    className="mr-2"
                  />
                  Production Volume
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 z-[1000]">
          <div className="bg-white rounded-lg shadow-lg p-3 max-w-[250px]">
            <div className="font-medium text-sm mb-2">
              {activeLayers.fsci ? "FSCI Classification" : "Production Volume"}
            </div>

            {activeLayers.fsci ? (
              <div className="space-y-1 text-xs">
                <div className="flex items-center">
                  <div
                    className="w-4 h-4 rounded mr-2"
                    style={{ backgroundColor: "#10B981" }}
                  ></div>
                  <span>Lumbung Pangan Primer (≥80)</span>
                </div>
                <div className="flex items-center">
                  <div
                    className="w-4 h-4 rounded mr-2"
                    style={{ backgroundColor: "#F59E0B" }}
                  ></div>
                  <span>Lumbung Pangan Sekunder (60-79)</span>
                </div>
                <div className="flex items-center">
                  <div
                    className="w-4 h-4 rounded mr-2"
                    style={{ backgroundColor: "#EF4444" }}
                  ></div>
                  <span>Lumbung Pangan Tersier (40-59)</span>
                </div>
                <div className="flex items-center">
                  <div
                    className="w-4 h-4 rounded mr-2"
                    style={{ backgroundColor: "#6B7280" }}
                  ></div>
                  <span>No Data (&lt;40)</span>
                </div>
              </div>
            ) : (
              <div className="space-y-1 text-xs">
                <div className="flex items-center">
                  <div
                    className="w-4 h-4 rounded mr-2"
                    style={{ backgroundColor: "#1E3A8A" }}
                  ></div>
                  <span>Very High (≥300K tons)</span>
                </div>
                <div className="flex items-center">
                  <div
                    className="w-4 h-4 rounded mr-2"
                    style={{ backgroundColor: "#3B82F6" }}
                  ></div>
                  <span>High (200-299K tons)</span>
                </div>
                <div className="flex items-center">
                  <div
                    className="w-4 h-4 rounded mr-2"
                    style={{ backgroundColor: "#60A5FA" }}
                  ></div>
                  <span>Medium (100-199K tons)</span>
                </div>
                <div className="flex items-center">
                  <div
                    className="w-4 h-4 rounded mr-2"
                    style={{ backgroundColor: "#93C5FD" }}
                  ></div>
                  <span>Low (50-99K tons)</span>
                </div>
                <div className="flex items-center">
                  <div
                    className="w-4 h-4 rounded mr-2"
                    style={{ backgroundColor: "#E5E7EB" }}
                  ></div>
                  <span>Very Low (&lt;50K tons)</span>
                </div>
              </div>
            )}

            {/* Performance Legend for Kabupaten */}
            {activeLayers.kabupaten && (
              <div className="border-t mt-2 pt-2">
                <div className="font-medium text-xs mb-1">
                  Kabupaten Performance (Borders)
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex items-center">
                    <div
                      className="w-4 h-1 mr-2"
                      style={{ backgroundColor: "#10B981" }}
                    ></div>
                    <span>Overperforming</span>
                  </div>
                  <div className="flex items-center">
                    <div
                      className="w-4 h-1 mr-2"
                      style={{ backgroundColor: "#6366F1" }}
                    ></div>
                    <span>Aligned</span>
                  </div>
                  <div className="flex items-center">
                    <div
                      className="w-4 h-1 mr-2"
                      style={{ backgroundColor: "#F59E0B" }}
                    ></div>
                    <span>Underperforming</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Selection Info */}
        {(selectedKabupaten || selectedKecamatan) && (
          <div className="absolute top-4 left-4 z-[1000]">
            <div className="bg-white rounded-lg shadow-lg p-3 max-w-[300px]">
              <div className="font-medium text-sm mb-2">Selected Area</div>

              {selectedKabupaten && (
                <Badge variant="secondary" className="mr-2">
                  Kabupaten: {selectedKabupaten}
                </Badge>
              )}

              {selectedKecamatan && (
                <Badge variant="outline">Kecamatan: {selectedKecamatan}</Badge>
              )}

              <div className="text-xs text-gray-600 mt-2">
                Click on areas to view detailed information
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
export type { TwoLevelMapProps };
