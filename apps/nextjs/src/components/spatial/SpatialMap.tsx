"use client";
import { useEffect, useRef, useState, useMemo } from "react";
import dynamic from "next/dynamic";
import {
  getSuitabilityColor,
  getSuitabilityClass,
  getFeatureStyle,
  getHighlightStyle,
  ACEH_BOUNDS,
} from "@/lib/spatial-map.utils";
import type { SpatialAnalysisResponse } from "@/lib/fetch/files.fetch";
import "leaflet/dist/leaflet.css";
import type L from "leaflet";

// Dynamic imports for ALL leaflet components (consistency fix)
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

interface SpatialMapProps {
  data: SpatialAnalysisResponse | null;
  isLoading: boolean;
  onFeatureClick?: (feature: any) => void;
  onFeatureHover?: (feature: any) => void;
}

export function SpatialMap({
  data,
  isLoading,
  onFeatureClick,
  onFeatureHover,
}: SpatialMapProps) {
  const [isMounted, setIsMounted] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<any>(null);
  const geoJsonRef = useRef<any>(null);

  // 🔧 FIX 1: Generate unique map key to prevent container reuse
  const mapKey = useMemo(() => `spatial-map-${Date.now()}`, []);

  const mapRef = useRef<L.Map | null>(null);

  // 🔧 FIX 2: Process data with unique keys to prevent duplicate issues
  const processedData = useMemo(() => {
    if (!data || !data.features) return null;

    return {
      ...data,
      features: data.features.map((feature, index) => ({
        ...feature,
        id: `feature_${feature.properties.GID_3}_${index}`, // Ensure unique ID
      })),
    };
  }, [data]);

  // Ensure component only renders on client
  useEffect(() => {
    setIsMounted(true);

    // 🔧 FIX 3: Cleanup on unmount
    return () => {
      setIsMounted(false);
    };
  }, []);

  // Handle feature interactions
  const onEachFeature = (feature: any, layer: any) => {
    const props = feature.properties;

    // Popup content
    const popupContent = `
      <div class="p-3 min-w-64">
        <h3 class="font-bold text-lg mb-2">${props.NAME_3}</h3>
        <p class="text-sm text-gray-600 mb-3">${props.NAME_2} Regency</p>
        
        <div class="space-y-2">
          <div class="flex justify-between">
            <span class="font-medium">Suitability Score:</span>
            <span class="font-bold text-lg" style="color: ${getSuitabilityColor(
              props.suitability_score
            )}">
              ${props.suitability_score?.toFixed(1) || "N/A"}
            </span>
          </div>
          
          <div class="flex justify-between">
            <span>Classification:</span>
            <span class="font-medium">${
              props.classification ||
              getSuitabilityClass(props.suitability_score)
            }</span>
          </div>
          
          <div class="flex justify-between">
            <span>Risk Level:</span>
            <span class="capitalize">${props.overall_risk}</span>
          </div>
        </div>
        
        <hr class="my-3">
        
        <div class="space-y-1 text-sm">
          <h4 class="font-medium mb-2">Climate Averages</h4>
          <div class="flex justify-between">
            <span>🌡️ Temperature:</span>
            <span>${props.avg_temperature?.toFixed(1)}°C</span>
          </div>
          <div class="flex justify-between">
            <span>🌧️ Precipitation:</span>
            <span>${props.avg_precipitation?.toFixed(1)} mm/day</span>
          </div>
          <div class="flex justify-between">
            <span>💧 Humidity:</span>
            <span>${props.avg_humidity?.toFixed(1)}%</span>
          </div>
        </div>
        
        <hr class="my-3">
        
        <div class="space-y-1 text-sm">
          <h4 class="font-medium mb-2">Component Scores</h4>
          <div class="flex justify-between">
            <span>Temperature:</span>
            <span>${props.temperature_score?.toFixed(0)}%</span>
          </div>
          <div class="flex justify-between">
            <span>Precipitation:</span>
            <span>${props.precipitation_score?.toFixed(0)}%</span>
          </div>
          <div class="flex justify-between">
            <span>Humidity:</span>
            <span>${props.humidity_score?.toFixed(0)}%</span>
          </div>
        </div>
      </div>
    `;

    // Bind popup
    layer.bindPopup(popupContent, {
      maxWidth: 300,
      className: "custom-popup",
    });

    // Mouse events
    layer.on({
      mouseover: (e: any) => {
        const layer = e.target;
        layer.setStyle(getHighlightStyle);
        layer.bringToFront();

        // Callback for parent component
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

        // Callback for parent component
        if (onFeatureClick) {
          onFeatureClick(feature);
        }
      },
    });
  };

  // 🔧 FIX 4: Better loading state check
  if (!isMounted) {
    return (
      <div className="w-full h-full bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">🗺️</div>
          <p className="text-gray-600">Initializing map...</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="w-full h-full bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading map data...</p>
        </div>
      </div>
    );
  }      <MapContainer
        key={mapKey}
        center={[4.5, 96.5]} // Center of Aceh
        zoom={8}
        style={{ height: "100%", width: "100%" }}
        zoomControl={true}
        scrollWheelZoom={true}
      ></MapContainer>

  // No data state
  if (
    !processedData ||
    !processedData.features ||
    processedData.features.length === 0
  ) {
    return (
      <div className="w-full h-full bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">🗺️</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No Data Available
          </h3>
          <p className="text-sm text-gray-600">
            Run spatial analysis to view district suitability data
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      {/* 🔧 FIX 5: Add unique key to force clean remount */}
      <MapContainer
        key={mapKey}
        ref={mapRef}
        center={[4.5, 96.5]}
        zoom={8}
        style={{ height: "100%", width: "100%" }}
        zoomControl={true}
        scrollWheelZoom={true}
      >
        {/* Base tile layer */}
        <TileLayer
          key="base-tiles"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />

        {/* 🔧 FIX 7: GeoJSON with unique key and processed data */}
        <GeoJSON
          key={`geojson-${processedData.features.length}-${mapKey}`}
          ref={geoJsonRef}
          data={processedData}
          style={getFeatureStyle}
          onEachFeature={onEachFeature}
        />
      </MapContainer>

      {/* Map info overlay */}
      <div className="absolute top-4 left-4 bg-white bg-opacity-90 p-3 rounded-lg shadow-lg z-[1000]">
        <div className="text-sm">
          <div className="font-medium">Analyzed Districts</div>
          <div className="text-2xl font-bold text-blue-600">
            {processedData.metadata?.analyzed_districts ||
              processedData.features.length}
          </div>
          <div className="text-xs text-gray-600">
            {processedData.metadata?.analysis_date
              ? new Date(
                  processedData.metadata.analysis_date
                ).toLocaleDateString()
              : "Latest Analysis"}
          </div>
        </div>
      </div>

      {/* Selected feature info */}
      {selectedFeature && (
        <div className="absolute bottom-4 left-4 bg-white bg-opacity-95 p-3 rounded-lg shadow-lg z-[1000] max-w-xs">
          <div className="text-sm">
            <div className="font-medium">
              {selectedFeature.properties.NAME_3}
            </div>
            <div className="text-xs text-gray-600">
              {selectedFeature.properties.NAME_2}
            </div>
            <div className="mt-1">
              <span
                className="text-lg font-bold"
                style={{
                  color: getSuitabilityColor(
                    selectedFeature.properties.suitability_score
                  ),
                }}
              >
                {selectedFeature.properties.suitability_score?.toFixed(1)}
              </span>
              <span className="text-sm ml-2">
                {selectedFeature.properties.classification}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
