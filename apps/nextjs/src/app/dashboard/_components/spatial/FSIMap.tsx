"use client";
import { useEffect, useRef, useState, useMemo } from "react";
import dynamic from "next/dynamic";
import { useQuery } from "@tanstack/react-query";
import { getSpatialFSIAnalysis } from "@/lib/fetch/spatial.map.fetch";
import type {
  FSIAnalysisParams,
  SpatialAnalysisResponse,
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

interface FSIMapProps {
  analysisParams?: FSIAnalysisParams;
  level?: "kabupaten" | "kecamatan";
  onFeatureClick?: (feature: any) => void;
  onFeatureHover?: (feature: any) => void;
  className?: string;
}

function getFSIPerformanceLevel(feature: any): string {
  // Use the actual classification from Flask hybrid system
  return feature.properties.fsi_class || "No Data";
}

// FSI Color scheme function
function getFSIColor(feature: any): string {
  const fsiClass = feature.properties.fsi_class;
  // Map Flask classification to colors
  switch (fsiClass) {
    case "Sangat Tinggi":
      return "#22c55e"; // Green
    case "Tinggi":
      return "#84cc16"; // Lime
    case "Sedang":
      return "#f59e0b"; // Amber
    case "Rendah":
      return "#f97316"; // Orange
    case "Sangat Rendah":
      return "#ef4444"; // Red
    default:
      return "#9ca3af"; // Gray for "No Data"
  }
}

export function FSIMap({
  analysisParams,
  level,
  onFeatureClick,
  onFeatureHover,
  className,
}: FSIMapProps) {
  const [isMounted, setIsMounted] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<any>(null);
  const geoJsonRef = useRef<any>(null);
  const mapRef = useRef<L.Map | null>(null);

  // Default parameters
  const defaultParams: FSIAnalysisParams = {
    districts: "all",
    year_start: 2018,
    year_end: 2024,
    bps_start_year: 2018,
    bps_end_year: 2024,
    season: "all",
    aggregation: "mean",
    analysis_level: "both",
    include_bps_data: true,
  };
  const params = analysisParams || defaultParams;

  // Generate unique map key
  const mapKey = useMemo(() => `fsi-map-${Date.now()}`, []);

  // ✅ Fetch FSI analysis data (Updated from FSCI)
  const {
    data: analysisData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["fsi-spatial", params],
    queryFn: async () => {
      return await getSpatialFSIAnalysis(params);
    },
    refetchOnWindowFocus: false,
  });

  // Process data for mapping
  const processedData = useMemo(() => {
    if (!analysisData) return null;

    if (analysisData.type === "FeatureCollection" && analysisData.features) {
      console.log(`✅ FSI GeoJSON loaded:`, {
        // ✅ Updated logging
        featureCount: analysisData.features.length,
        analysisType: "food_security_index_spatial_analysis",
        avgFSI:
          analysisData.metadata.summary_statistics.average_scores.fsi_score,
      });

      return analysisData;
    }
    console.warn("❌ Invalid GeoJSON structure received:", analysisData);
    return null;
  }, [analysisData]);

  // ✅ Feature styling using FSI scores (Updated from FSCI)
  const getFeatureStyle = (feature: any) => {
    const isSelected =
      selectedFeature?.properties?.NAME_3 === feature.properties.NAME_3;

    return {
      fillColor: getFSIColor(feature), // ✅ Use Flask classification
      weight: isSelected ? 3 : 1,
      opacity: 1,
      color: isSelected ? "#1f2937" : "#ffffff",
      dashArray: isSelected ? "5, 5" : "",
      fillOpacity: 0.8,
    };
  };

  // Ensure component only renders on client side
  useEffect(() => {
    setIsMounted(true);
    return () => {
      setIsMounted(false);
    };
  }, []);

  // ✅ Handle feature interactions with FSI data (Updated from FSCI)
  const onEachFeature = (feature: any, layer: any) => {
    const props = feature.properties;

    const safeFSIScore =
      typeof props.fsi_score === "number" && !isNaN(props.fsi_score)
        ? props.fsi_score
        : 0;

    const regionName = props.NAME_3 || props.NAME_2 || "Unknown Region";

    // ✅ Use Flask classification instead of manual calculation
    const flaskClassification = props.fsi_class || "No Data";

    const safeNaturalResources = props.natural_resources_score || 0;
    const safeAvailability = props.availability_score || 0;

    // ✅ FIXED: Enhanced popup content for FSI data
    const popupContent = `
    <div class="p-3 min-w-[280px]">
      <div class="font-semibold text-lg text-gray-900 mb-2">
        ${regionName}
      </div>
      <div class="text-xs text-gray-600 mb-3">
        Kecamatan Level • FSI Analysis (Flask Hybrid)
      </div>
      
      <div class="space-y-3">
        <!-- FSI Score Display -->
        <div class="text-center p-3 rounded-lg" style="background-color: ${getFSIColor(
          feature
        )}20;">
          <div class="text-xs text-gray-600 mb-1">Food Security Index</div>
          <div class="text-2xl font-bold" style="color: ${getFSIColor(
            feature
          )}">
            ${safeFSIScore.toFixed(1)}
          </div>
          <div class="text-sm font-medium mt-1">${flaskClassification}</div>
          <div class="text-xs text-gray-500 mt-1">Flask Hybrid Classification</div>
        </div>
        
        <!-- FSI Components (60/40 weighting) -->
        <div class="grid grid-cols-2 gap-3">
          <div class="text-center p-2 bg-blue-50 rounded">
            <div class="text-xs text-gray-600">Sumber Daya Alam</div>
            <div class="text-lg font-semibold text-blue-700">${safeNaturalResources.toFixed(
              1
            )}</div>
            <div class="text-xs text-gray-500">60% bobot</div>
          </div>
          <div class="text-center p-2 bg-green-50 rounded">
            <div class="text-xs text-gray-600">Ketersediaan</div>
            <div class="text-lg font-semibold text-green-700">${safeAvailability.toFixed(
              1
            )}</div>
            <div class="text-xs text-gray-500">40% bobot</div>
          </div>
        </div>
        
        <!-- Climate Data -->
        <div class="border-t pt-2 mt-2">
          <div class="text-xs text-gray-600 mb-2">Data Iklim:</div>
          <div class="grid grid-cols-3 gap-2 text-xs">
            <div class="text-center">
              <div class="text-gray-500">Suhu</div>
              <div class="font-medium">${(props.avg_temperature || 0).toFixed(
                1
              )}°C</div>
            </div>
            <div class="text-center">
              <div class="text-gray-500">Curah Hujan</div>
              <div class="font-medium">${(props.avg_precipitation || 0).toFixed(
                1
              )}mm</div>
            </div>
            <div class="text-center">
              <div class="text-gray-500">Kelembapan</div>
              <div class="font-medium">${(props.avg_humidity || 0).toFixed(
                1
              )}%</div>
            </div>
          </div>
        </div>
        
        
        ${
          props.nasa_match
            ? `
          <div class="text-xs text-blue-600 mt-2">
            NASA Location: ${props.nasa_match}
          </div>
        `
            : ""
        }
      </div>
    </div>
  `;

    layer.bindPopup(popupContent, {
      maxWidth: 320,
      className: "custom-fsi-popup", // ✅ Updated class name
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
        setSelectedFeature(feature); // ✅ Keep for popup/tooltip
        if (onFeatureClick) {
          onFeatureClick(feature);
        }
      },
    });
  };

  // Loading state
  if (!isMounted) {
    return (
      <div
        className={`w-full h-full bg-gray-100 flex items-center justify-center ${className}`}
      >
        <div className="text-center">
          <div className="text-4xl mb-4">🗺️</div>
          <p className="text-gray-600">Initializing FSI map...</p>{" "}
          {/* ✅ Updated text */}
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
          <p className="text-gray-600">Loading FSI analysis data...</p>{" "}
          {/* ✅ Updated text */}
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
            Failed to load FSI analysis data
          </p>{" "}
          {/* ✅ Updated text */}
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
            No FSI Data Available
          </h3>{" "}
          {/* ✅ Updated text */}
          <p className="text-sm text-gray-600">
            Adjust analysis parameters to view regional FSI data
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`w-full h-full relative ${className}`}>
      {/* ✅ FSI Map Container (Updated from FSCI) */}
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
          key="fsi-base-tiles" // ✅ Updated key
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />

        {/* ✅ GeoJSON with FSI data (Updated from FSCI) */}
        <GeoJSON
          key={`fsi-geojson-${processedData.features.length}-${mapKey}`} // ✅ Updated key
          ref={geoJsonRef}
          data={processedData}
          style={getFeatureStyle}
          onEachFeature={onEachFeature}
        />
      </MapContainer>

      {/* ✅ FSI Analysis Info Overlay (Updated from FSCI) */}
      <div className="absolute top-4 left-4 bg-white bg-opacity-95 p-4 rounded-lg shadow-lg z-[1000]">
        <div className="text-sm space-y-1">
          <div className="font-semibold text-gray-900">FSI Analysis</div>{" "}
          {/* ✅ Updated */}
          <div className="text-xs text-gray-600">Kecamatan Level</div>
          <div className="text-2xl font-bold text-blue-600">
            {analysisData?.metadata?.analyzed_districts || 0}
          </div>
          <div className="text-xs text-gray-600">regions analyzed</div>
          <div className="text-xs text-gray-500 pt-1 border-t">
            Avg FSI:{" "}
            {analysisData?.metadata?.summary_statistics?.average_scores?.fsi_score?.toFixed(
              1
            ) || "N/A"}
          </div>
        </div>
      </div>

      <div className="absolute top-4 right-4 bg-white bg-opacity-95 p-3 rounded-lg shadow-lg z-[1000]">
        <div className="text-sm space-y-2">
          <div className="font-semibold text-gray-900">
            FSI Performance (Flask Hybrid)
          </div>
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: "#22c55e" }}
              ></div>
              <span className="text-xs">Sangat Tinggi</span>
            </div>
            <div className="flex items-center space-x-2">
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: "#84cc16" }}
              ></div>
              <span className="text-xs">Tinggi</span>
            </div>
            <div className="flex items-center space-x-2">
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: "#f59e0b" }}
              ></div>
              <span className="text-xs">Sedang</span>
            </div>
            <div className="flex items-center space-x-2">
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: "#f97316" }}
              ></div>
              <span className="text-xs">Rendah</span>
            </div>
            <div className="flex items-center space-x-2">
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: "#ef4444" }}
              ></div>
              <span className="text-xs">Sangat Rendah</span>
            </div>
          </div>
          <div className="text-xs text-gray-500 pt-1 border-t">
            BPS-Calibrated Classification
          </div>
        </div>
      </div>

      {/* ✅ Selected Feature Info Panel with FSI Components (Updated from FSCI) */}
      {selectedFeature && (
        <div className="absolute bottom-4 left-4 bg-white bg-opacity-95 p-4 rounded-lg shadow-lg z-[1000] max-w-sm">
          <div className="text-sm space-y-2">
            <div className="font-semibold text-gray-900">
              {selectedFeature.properties.NAME_3 ||
                selectedFeature.properties.NAME_2 ||
                "Unknown Region"}
            </div>
            <div className="text-xs text-gray-600">
              Kecamatan Level Analysis (Flask Hybrid)
            </div>

            <div className="flex items-center justify-between pt-2 border-t">
              <span className="text-gray-700">FSI Score:</span>
              <span
                className="text-xl font-bold"
                style={{
                  color: getFSIColor(selectedFeature), // ✅ Use Flask color
                }}
              >
                {typeof selectedFeature.properties.fsi_score === "number" &&
                !isNaN(selectedFeature.properties.fsi_score)
                  ? selectedFeature.properties.fsi_score.toFixed(1)
                  : "0.0"}
              </span>
            </div>

            <div className="text-xs">
              <span className="font-medium">
                {selectedFeature.properties.fsi_class || "No Data"}{" "}
                {/* ✅ Use Flask classification */}
              </span>
              <span className="text-gray-500 ml-2">(Flask Hybrid)</span>
            </div>

            {/* ✅ FSI Component Scores (Updated from PCI/PSI/CRS) */}
            <div className="grid grid-cols-2 gap-2 pt-2 border-t text-xs">
              <div className="text-center p-2 bg-blue-50 rounded">
                <div className="text-gray-500">Sumber Daya Alam</div>
                <div className="font-medium text-blue-700">
                  {(
                    selectedFeature.properties.natural_resources_score || 0
                  ).toFixed(1)}
                </div>
                <div className="text-xs text-gray-400">60%</div>
              </div>
              <div className="text-center p-2 bg-green-50 rounded">
                <div className="text-gray-500">Ketersediaan</div>
                <div className="font-medium text-green-700">
                  {(selectedFeature.properties.availability_score || 0).toFixed(
                    1
                  )}
                </div>
                <div className="text-xs text-gray-400">40%</div>
              </div>
            </div>

            <div className="text-xs text-gray-600 space-y-1 pt-2 border-t">
              <div>
                Area: {selectedFeature.properties.area_km2?.toFixed(1) || "N/A"}{" "}
                km²
              </div>

              {selectedFeature.properties.nasa_match && (
                <div className="text-blue-600">
                  NASA: {selectedFeature.properties.nasa_match}
                </div>
              )}

              <div className="grid grid-cols-3 gap-1 text-xs mt-2">
                <div className="text-center">
                  <div className="text-gray-400">Temp</div>
                  <div>
                    {(selectedFeature.properties.avg_temperature || 0).toFixed(
                      1
                    )}
                    °C
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-gray-400">Rain</div>
                  <div>
                    {(
                      selectedFeature.properties.avg_precipitation || 0
                    ).toFixed(1)}
                    mm
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-gray-400">Humid</div>
                  <div>
                    {(selectedFeature.properties.avg_humidity || 0).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
