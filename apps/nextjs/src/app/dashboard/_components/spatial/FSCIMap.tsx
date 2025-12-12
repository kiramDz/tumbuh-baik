"use client";
import { useEffect, useRef, useState, useMemo } from "react";
import dynamic from "next/dynamic";
import { useQuery } from "@tanstack/react-query";
import { getTwoLevelAnalysis } from "@/lib/fetch/spatial.map.fetch";
import type {
  TwoLevelAnalysisParams,
  TwoLevelAnalysisResponse,
} from "@/lib/fetch/spatial.map.fetch";
import "leaflet/dist/leaflet.css";
import type L from "leaflet";

// Dynamic imports for All leaflet components
const MapContainer = dynamic(
  () => import("react-leaflet").then((mod) => mod.MapContainer),
  { ssr: false }
);

const TileLayer = dynamic(
  () => import("react-leaflet").then((mod) => mod.TileLayer),
  { ssr: false }
);

const GeoJSON = dynamic(
  () => import("react-leaflet").then((mod) => mod.GeoJSON),
  { ssr: false }
);

interface FSCIMapProps {
  analysisParams?: TwoLevelAnalysisParams;
  level?: "kabupaten" | "kecamatan";
  onFeatureClick?: (feature: any) => void;
  onFeatureHover?: (feature: any) => void;
  selectedRegion?: string | null;
  className?: string;
}

function getFSCIClassification(score: number): string {
  if (score >= 75) return "excellent";
  if (score >= 60) return "good";
  if (score >= 45) return "fair";
  return "poor";
}

function getFSCIPerformanceLevel(score: number): string {
  if (score >= 75) return "Excellent Performance";
  if (score >= 60) return "Good Performance";
  if (score >= 45) return "Fair Performance";
  return "Needs Improvement";
}

// FSCI Color scheme function
function getFSCIColor(score: number): string {
  if (score >= 75) return "#059669"; // Green (Excellent)
  if (score >= 60) return "#3B82F6"; // Blue (Good)
  if (score >= 45) return "#F59E0B"; // Orange (Fair)
  return "#DC2626"; // Red (Poor)
}

export function FSCIMap({
  analysisParams,
  level = "kabupaten",
  onFeatureClick,
  onFeatureHover,
  selectedRegion,
  className,
}: FSCIMapProps) {
  const [isMounted, setIsMounted] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<any>(null);
  const geoJsonRef = useRef<any>(null);
  const mapRef = useRef<L.Map | null>(null);

  // Default parameters
  const defaultParams: TwoLevelAnalysisParams = {
    year_start: 2018,
    year_end: 2024,
    bps_start_year: 2018,
    bps_end_year: 2024,
    season: "all",
    aggregation: "mean",
    districts: "all",
  };
  const params = analysisParams || defaultParams;

  // Generate unique map key to prevent container reuse
  const mapKey = useMemo(() => `fsci-map-${level}-${Date.now()}`, [level]);

  // Fetch FSCI analysis data
  const {
    data: analysisData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["fsci-boundaries", params, level],
    queryFn: async () => {
      const stringParams = {
        level,
        year_start: (params.year_start || 2018).toString(),
        year_end: (params.year_end || 2024).toString(),
        bps_start_year: (params.bps_start_year || 2018).toString(),
        bps_end_year: (params.bps_end_year || 2024).toString(),
        season: params.season || "all",
        aggregation: params.aggregation || "mean",
        districts: params.districts || "all",
      };

      const searchParams = new URLSearchParams(stringParams);

      // ✅ FIX: Use localhost:5001 to match your Flask server
      const response = await fetch(
        `http://localhost:5001/api/v1/two-level/analysis-with-boundaries?${searchParams}`
      );

      if (!response.ok) throw new Error("Failed to fetch FSCI boundaries");
      return response.json();
    },
    refetchOnWindowFocus: false,
  });

  // Process data for mapping with keys
  const processedData = useMemo(() => {
    if (!analysisData) return null;

    if (analysisData.type === "FeatureCollection" && analysisData.features) {
      console.log(`✅ Direct GeoJSON usage - ${level} level:`, {
        featureCount: analysisData.features.length,
        geometryType:
          analysisData.metadata?.geometry_type || "Polygon/MultiPolygon",
        analysisType:
          analysisData.metadata?.analysis_type || "fsci_with_boundaries",
        dataIntegration: analysisData.metadata?.data_integration || "merged",
      });

      return analysisData;
    }
    console.warn("❌ Invalid GeoJSON structure received:", analysisData);
    return null;
  }, [analysisData, level]);

  // Feature styling
  const getFeatureStyle = (feature: any) => {
    const score = feature.properties.fsci_score || 0;
    const isSelected =
      selectedRegion ===
      (feature.properties.NAME_3 || feature.properties.NAME_2);

    return {
      fillColor: getFSCIColor(score),
      weight: isSelected ? 3 : 1,
      opacity: 1,
      color: isSelected ? "#1f2937" : "#ffffff",
      dashArray: isSelected ? "5, 5" : "",
      fillOpacity: 0.8,
    };
  };

  // Highlight style for hover
  const getHighlightStyle = {
    weight: 3,
    color: "#1f2937",
    dashArray: "",
    fillOpacity: 0.9,
  };

  // Ensure component only renders on client side
  useEffect(() => {
    setIsMounted(true);
    return () => {
      setIsMounted(false);
    };
  }, []);

  // Handle feature interactions
  const onEachFeature = (feature: any, layer: any) => {
    const props = feature.properties;

    // ✅ Safe property access
    const safeScore =
      typeof props.fsci_score === "number" && !isNaN(props.fsci_score)
        ? props.fsci_score
        : 0;

    const regionName = props.NAME_3 || props.NAME_2 || "Unknown Region";
    const safePerformance =
      props.performance_level || getFSCIPerformanceLevel(safeScore);
    const safeProduction = props.production_display || "No Production Data";
    const safeCorrelation = props.correlation_display || "N/A";

    // ✅ Enhanced popup content for boundaries data
    const popupContent = `
    <div class="p-3 min-w-[250px]">
      <div class="font-semibold text-lg text-gray-900 mb-2">
        ${regionName}
      </div>
      <div class="text-xs text-gray-600 mb-2">
        ${level === "kabupaten" ? "Kabupaten" : "Kecamatan"} Level
      </div>
      
      <div class="space-y-2">
        <div class="flex justify-between items-center">
          <span class="text-sm text-gray-600">FSCI Score:</span>
          <span class="font-semibold text-lg" style="color: ${getFSCIColor(
            safeScore
          )}">
            ${safeScore.toFixed(1)}
          </span>
        </div>
        
        <div class="flex justify-between items-center">
          <span class="text-sm text-gray-600">Performance:</span>
          <span class="text-sm font-medium">${safePerformance}</span>
        </div>
        
        <div class="border-t pt-2 mt-2">
          <div class="flex justify-between items-center">
            <span class="text-xs text-gray-600">${
              level === "kabupaten" ? "Rice Production" : "Climate Analysis"
            }:</span>
            <span class="text-sm font-medium">${safeProduction}</span>
          </div>
          
          ${
            level === "kabupaten"
              ? `
            <div class="flex justify-between items-center">
              <span class="text-xs text-gray-600">Climate Correlation:</span>
              <span class="text-sm font-medium">${safeCorrelation}</span>
            </div>
          `
              : ""
          }
        </div>
        
        <div class="text-xs text-gray-500 mt-2">
          Area: ${props.area_km2?.toFixed(1) || "N/A"} km²
        </div>
        
        ${
          level === "kecamatan" && props.nasa_match
            ? `
          <div class="text-xs text-blue-600 mt-2">
            NASA Match: ${props.nasa_match}
          </div>
        `
            : ""
        }
        
        ${
          props.investment_recommendation
            ? `
          <div class="text-xs text-green-600 mt-2">
            ${props.investment_recommendation}
          </div>
        `
            : ""
        }
      </div>
    </div>
  `;

    layer.bindPopup(popupContent, {
      maxWidth: 300,
      className: "custom-fsci-popup",
    });

    layer.on({
      mouseover: (e: any) => {
        const layer = e.target;
        layer.setStyle({
          weight: 3,
          color: "#1f2937",
          fillOpacity: 0.9,
        });
        layer.bringToFront();

        if (onFeatureHover) {
          onFeatureHover(feature);
        }
      },
      mouseout: (e: any) => {
        const layer = e.target;
        if (geoJsonRef.current) {
          geoJsonRef.current.resetStyle(layer);
        }
      },
      click: (e: any) => {
        setSelectedFeature(feature);
        if (onFeatureClick) {
          onFeatureClick(feature);
        }
      },
    });
  };

  // Loading state check
  if (!isMounted) {
    return (
      <div
        className={`w-full h-full bg-gray-100 flex items-center justify-center ${className}`}
      >
        <div className="text-center">
          <div className="text-4xl mb-4">🗺️</div>
          <p className="text-gray-600">Initializing FSCI map...</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div
        className={`w-full h-full bg-gray-100 flex items-center justify-center ${className}`}
      >
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading FSCI analysis data...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error || !analysisData) {
    return (
      <div
        className={`w-full h-full bg-gray-100 flex items-center justify-center ${className}`}
      >
        <div className="text-center">
          <div className="text-4xl mb-4 text-red-500">⚠️</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Error Loading Map
          </h3>
          <p className="text-sm text-gray-600">
            Failed to load FSCI analysis data
          </p>
        </div>
      </div>
    );
  }

  // No data state
  if (!processedData || processedData.features.length === 0) {
    return (
      <div
        className={`w-full h-full bg-gray-100 flex items-center justify-center ${className}`}
      >
        <div className="text-center">
          <div className="text-4xl mb-4">🗺️</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No FSCI Data Available
          </h3>
          <p className="text-sm text-gray-600">
            Adjust analysis parameters to view regional FSCI data
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`w-full h-full relative ${className}`}>
      {/* FSCI Map Container */}
      <MapContainer
        key={mapKey}
        ref={mapRef}
        center={[4.695135, 96.749397]}
        zoom={8}
        style={{ height: "100%", width: "100%" }}
        zoomControl={true}
        scrollWheelZoom={true}
      >
        {/* Base tile layer */}
        <TileLayer
          key="fsci-base-tiles"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />

        {/* GeoJSON with FSCI data */}
        <GeoJSON
          key={`fsci-geojson-${processedData.features.length}-${mapKey}`}
          ref={geoJsonRef}
          data={processedData}
          style={getFeatureStyle}
          onEachFeature={onEachFeature}
        />
      </MapContainer>

      {/* FSCI Analysis Info Overlay */}
      <div className="absolute top-4 left-4 bg-white bg-opacity-95 p-4 rounded-lg shadow-lg z-[1000]">
        <div className="text-sm space-y-1">
          <div className="font-semibold text-gray-900">FSCI Analysis</div>
          <div className="text-xs text-gray-600">
            {level === "kabupaten" ? "Kabupaten" : "Kecamatan"} Level
          </div>
          <div className="text-2xl font-bold text-blue-600">
            {analysisData?.metadata?.feature_count || 0}
          </div>
          <div className="text-xs text-gray-600">regions analyzed</div>
          <div className="text-xs text-gray-500 pt-1 border-t">
            Real Administrative Boundaries
          </div>
          <div className="text-xs text-gray-500">
            {analysisData?.metadata?.geometry_type || "Polygon/MultiPolygon"}
          </div>
        </div>
      </div>

      {/* FSCI Color Legend */}
      <div className="absolute top-4 right-4 bg-white bg-opacity-95 p-3 rounded-lg shadow-lg z-[1000]">
        <div className="text-sm space-y-2">
          <div className="font-semibold text-gray-900">FSCI Performance</div>
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: "#059669" }}
              ></div>
              <span className="text-xs">Excellent (75+)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: "#3B82F6" }}
              ></div>
              <span className="text-xs">Good (60-74)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: "#F59E0B" }}
              ></div>
              <span className="text-xs">Fair (45-59)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: "#DC2626" }}
              ></div>
              <span className="text-xs">Poor (&lt;45)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Selected Feature Info Panel */}
      {selectedFeature && (
        <div className="absolute bottom-4 left-4 bg-white bg-opacity-95 p-4 rounded-lg shadow-lg z-[1000] max-w-sm">
          <div className="text-sm space-y-2">
            <div className="font-semibold text-gray-900">
              {selectedFeature.properties.NAME_3 ||
                selectedFeature.properties.NAME_2 ||
                "Unknown Region"}
            </div>
            <div className="text-xs text-gray-600 capitalize">
              {level} Level Analysis
            </div>

            <div className="flex items-center justify-between pt-2 border-t">
              <span className="text-gray-700">FSCI Score:</span>
              <span
                className="text-xl font-bold"
                style={{
                  color: getFSCIColor(
                    selectedFeature.properties.fsci_score || 0
                  ),
                }}
              >
                {typeof selectedFeature.properties.fsci_score === "number" &&
                !isNaN(selectedFeature.properties.fsci_score)
                  ? selectedFeature.properties.fsci_score.toFixed(1)
                  : "0.0"}
              </span>
            </div>

            <div className="text-xs">
              <span className="font-medium">
                {selectedFeature.properties.performance_level ||
                  getFSCIPerformanceLevel(
                    selectedFeature.properties.fsci_score || 0
                  )}
              </span>
            </div>

            {/* Component Scores */}
            <div className="grid grid-cols-3 gap-2 pt-2 border-t text-xs">
              <div className="text-center">
                <div className="text-gray-500">PCI</div>
                <div className="font-medium">
                  {(selectedFeature.properties.pci_score || 0).toFixed(1)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-gray-500">PSI</div>
                <div className="font-medium">
                  {(selectedFeature.properties.psi_score || 0).toFixed(1)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-gray-500">CRS</div>
                <div className="font-medium">
                  {(selectedFeature.properties.crs_score || 0).toFixed(1)}
                </div>
              </div>
            </div>

            <div className="text-xs text-gray-600 space-y-1 pt-2 border-t">
              <div>
                Area: {selectedFeature.properties.area_km2?.toFixed(1) || "N/A"}{" "}
                km²
              </div>

              {level === "kecamatan" &&
                selectedFeature.properties.nasa_match && (
                  <div className="text-blue-600">
                    NASA: {selectedFeature.properties.nasa_match}
                  </div>
                )}

              {selectedFeature.properties.investment_recommendation && (
                <div className="text-green-600">
                  {selectedFeature.properties.investment_recommendation}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
