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
    queryKey: ["fsci-map-analysis", params, level],
    queryFn: () => getTwoLevelAnalysis(params),
    refetchOnWindowFocus: false,
  });

  // Process data for mapping with keys
  const processedData = useMemo(() => {
    if (!analysisData) return null;

    const sourceData =
      level === "kabupaten"
        ? analysisData.level_2_kabupaten_analysis?.data || []
        : analysisData.level_1_kecamatan_analysis?.data || [];

    if (sourceData.length === 0) return null;

    // Helper function to safely get values
    const getValue = (item: any, ...keys: string[]): number => {
      for (const key of keys) {
        const value = item[key];
        if (typeof value === "number" && !isNaN(value)) return value;
      }
      return 0;
    };

    // ✅ Transform analysis data with REAL coordinates from two-level API
    const features = sourceData
      .map((item: any, index: number) => {
        // Get FSCI score based on level
        const fsciScore =
          level === "kabupaten"
            ? item.aggregated_fsci_score
            : item.fsci_analysis?.fsci_score;

        // ✅ NEW: Extract real coordinates from two-level API data
        const coordinates =
          level === "kabupaten"
            ? [
                // For kabupaten: use centroid or aggregated coordinates
                item.centroid_longitude ||
                  item.aggregated_longitude ||
                  getValue(item, "longitude", "lng") ||
                  96.5, // Fallback to Aceh center
                item.centroid_latitude ||
                  item.aggregated_latitude ||
                  getValue(item, "latitude", "lat") ||
                  4.5, // Fallback to Aceh center
              ]
            : [
                // For kecamatan: use NASA location or direct coordinates
                item.nasa_location_longitude ||
                  item.longitude ||
                  getValue(item, "lng", "nasa_lng") ||
                  96.5, // Fallback to Aceh center
                item.nasa_location_latitude ||
                  item.latitude ||
                  getValue(item, "lat", "nasa_lat") ||
                  4.5, // Fallback to Aceh center
              ];

        const production =
          level === "kabupaten"
            ? item.bps_validation?.latest_production_tons ||
              item.latest_production_tons ||
              0
            : 0; // Level 1 doesn't have production data

        const correlation =
          level === "kabupaten"
            ? item.climate_production_correlation ||
              getValue(item, "correlation") ||
              0
            : 0; // Level 1 doesn't have correlation

        const regionName =
          level === "kabupaten" ? item.kabupaten_name : item.kecamatan_name;

        if (!regionName || fsciScore === 0) return null;

        return {
          type: "Feature" as const,
          id: `fsci_feature_${item.id || index}`,
          properties: {
            // Basic identification
            NAME_2:
              level === "kabupaten" ? item.kabupaten_name : item.kabupaten_name,
            NAME_3:
              level === "kabupaten" ? item.kabupaten_name : item.kecamatan_name,
            GID_3: item.id || `${level}_${index}`,
            region_type: level,

            // ✅ NEW: Enhanced FSCI component data
            fsci_score: fsciScore,
            pci_score:
              level === "kabupaten"
                ? item.aggregated_pci_score || getValue(item, "pci_score") || 0
                : item.fsci_analysis?.pci?.pci_score ||
                  getValue(item, "pci_score") ||
                  0,
            psi_score:
              level === "kabupaten"
                ? item.aggregated_psi_score || getValue(item, "psi_score") || 0
                : item.fsci_analysis?.psi?.psi_score ||
                  getValue(item, "psi_score") ||
                  0,
            crs_score:
              level === "kabupaten"
                ? item.aggregated_crs_score || getValue(item, "crs_score") || 0
                : item.fsci_analysis?.crs?.crs_score ||
                  getValue(item, "crs_score") ||
                  0,

            area_km2:
              level === "kabupaten"
                ? item.total_area_km2 || getValue(item, "area_km2") || 0
                : item.area_km2 || getValue(item, "area") || 0,
            area_weight: item.area_weight || 0,

            // Production and validation data
            production_tons: production,
            climate_correlation: correlation,

            investment_recommendation:
              level === "kabupaten"
                ? item.investment_recommendation || "moderate"
                : item.investment_recommendation || "low",
            performance_gap_category:
              level === "kabupaten"
                ? item.performance_gap_category || "aligned"
                : "not_applicable",

            ...(level === "kabupaten" && {
              bps_data_available: !!item.bps_validation,
              production_trend:
                item.production_trend ||
                item.bps_validation?.production_trend ||
                "stable",
              data_coverage_years:
                item.data_coverage_years ||
                item.bps_validation?.data_coverage_years ||
                0,
              efficiency_score: item.production_efficiency_score || 0,
            }),

            // Classification and display
            fsci_classification: getFSCIClassification(fsciScore),
            performance_level: getFSCIPerformanceLevel(fsciScore),

            production_display:
              production > 0
                ? `${(production / 1000).toFixed(1)}K tons`
                : "No data",
            correlation_display:
              correlation > 0 ? correlation.toFixed(3) : "N/A",
            area_display:
              level === "kabupaten"
                ? `${((item.total_area_km2 || 0) / 1000).toFixed(1)}K km²`
                : `${(item.area_km2 || 0).toFixed(1)} km²`,

            component_summary: {
              pci:
                level === "kabupaten"
                  ? item.aggregated_pci_score
                  : item.fsci_analysis?.pci?.pci_score,
              psi:
                level === "kabupaten"
                  ? item.aggregated_psi_score
                  : item.fsci_analysis?.psi?.psi_score,
              crs:
                level === "kabupaten"
                  ? item.aggregated_crs_score
                  : item.fsci_analysis?.crs?.crs_score,
            },
          },
          geometry: {
            type: "Point" as const,
            coordinates: coordinates as [number, number],
          },
        };
      })
      .filter(
        (feature): feature is NonNullable<typeof feature> => feature !== null
      );

    return {
      type: "FeatureCollection" as const,
      features,
      metadata: {
        total_regions: features.length,
        analysis_date: new Date().toISOString(),
        level_analyzed: level,
        avg_fsci:
          features.length > 0
            ? features.reduce((sum, f) => sum + f.properties.fsci_score, 0) /
              features.length
            : 0,
        avg_pci:
          features.length > 0
            ? features.reduce(
                (sum, f) => sum + (f.properties.pci_score || 0),
                0
              ) / features.length
            : 0,
        avg_psi:
          features.length > 0
            ? features.reduce(
                (sum, f) => sum + (f.properties.psi_score || 0),
                0
              ) / features.length
            : 0,
        avg_crs:
          features.length > 0
            ? features.reduce(
                (sum, f) => sum + (f.properties.crs_score || 0),
                0
              ) / features.length
            : 0,
        coordinate_source:
          level === "kabupaten" ? "centroid_aggregated" : "nasa_locations",
        regions_with_production: features.filter(
          (f) => f.properties.production_tons > 0
        ).length,
      },
    };
  }, [analysisData, level]);

  // Feature styling
  const getFeatureStyle = (feature: any) => {
    const score = feature.properties.fsci_score;
    const isSelected = selectedRegion === feature.properties.NAME_2;

    return {
      fillColor: getFSCIColor(score),
      weight: isSelected ? 3 : 1,
      opacity: 1,
      color: isSelected ? "#1f2937" : "#ffffff",
      dashArray: isSelected ? "5, 5" : "",
      fillOpacity: 0.7,
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

    // Popup content with FSCI data
    const popupContent = `
      <div class="p-3 min-w-[250px]">
        <div class="font-semibold text-lg text-gray-900 mb-2">
          ${props.NAME_2}
        </div>
        
        <div class="space-y-2">
          <div class="flex justify-between items-center">
            <span class="text-sm text-gray-600">FSCI Score:</span>
            <span class="font-semibold text-lg" style="color: ${getFSCIColor(
              props.fsci_score
            )}">
              ${props.fsci_score.toFixed(1)}
            </span>
          </div>
          
          <div class="flex justify-between items-center">
            <span class="text-sm text-gray-600">Performance:</span>
            <span class="text-sm font-medium">${props.performance_level}</span>
          </div>
          
          <div class="border-t pt-2 mt-2">
            <div class="flex justify-between items-center">
              <span class="text-xs text-gray-600">Wheat Production:</span>
              <span class="text-sm font-medium">${
                props.production_display
              }</span>
            </div>
            
            <div class="flex justify-between items-center">
              <span class="text-xs text-gray-600">Climate Correlation:</span>
              <span class="text-sm font-medium">${
                props.correlation_display
              }</span>
            </div>
          </div>
          
          <div class="text-xs text-gray-500 mt-2">
            Classification: ${props.fsci_classification}
          </div>
        </div>
      </div>
    `;

    // Bind popup
    layer.bindPopup(popupContent, {
      maxWidth: 300,
      className: "custom-fsci-popup",
    });

    // Mouse events
    layer.on({
      mouseover: (e: any) => {
        const layer = e.target;
        layer.setStyle(getHighlightStyle);
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
        center={[-2.5, 118]} // Center of Indonesia
        zoom={5}
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
            {processedData.metadata?.total_regions || 0}
          </div>
          <div className="text-xs text-gray-600">regions analyzed</div>
          <div className="text-xs text-gray-500 pt-1 border-t">
            Avg FSCI: {processedData.metadata?.avg_fsci?.toFixed(1) || "N/A"}
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
              {selectedFeature.properties.NAME_2}
            </div>
            <div className="text-xs text-gray-600 capitalize">
              {selectedFeature.properties.region_type}
            </div>

            <div className="flex items-center justify-between pt-2 border-t">
              <span className="text-gray-700">FSCI Score:</span>
              <span
                className="text-xl font-bold"
                style={{
                  color: getFSCIColor(selectedFeature.properties.fsci_score),
                }}
              >
                {selectedFeature.properties.fsci_score.toFixed(1)}
              </span>
            </div>

            <div className="text-xs">
              <span className="font-medium">
                {selectedFeature.properties.performance_level}
              </span>
            </div>

            <div className="text-xs text-gray-600 space-y-1 pt-2 border-t">
              <div>
                Production: {selectedFeature.properties.production_display}
              </div>
              <div>
                Climate Correlation:{" "}
                {selectedFeature.properties.correlation_display}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
