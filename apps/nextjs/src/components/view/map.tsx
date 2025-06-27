"use client";
import "mapbox-gl/dist/mapbox-gl.css";
import { useEffect, useState } from "react";
import ReactMapGL, { Layer, LayerProps, Source } from "react-map-gl";
import { Card } from "../ui/card";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { useSearchParams } from "next/navigation";
import { DEFAULT_LOCATION } from "@/lib/config";
import { useTheme } from "next-themes";

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN;
const OPENWEATHERMAP_TOKEN = process.env.NEXT_PUBLIC_OPENWEATHER_API_KEY;

const weatherTiles = [
  { label: "Temperatur (°C)", code: "temp_new" },
  { label: "Presipitasi (mm)", code: "precipitation_new" },
  { label: "Angin (m/s)", code: "wind_new" },
  { label: "Kelembaban (%)", code: "clouds_new" },
  { label: "Tekanan (hPa)", code: "pressure_new" },
];

// Improved weather layer configuration
const weatherLayer: LayerProps = {
  id: "weatherLayer",
  type: "raster",
  paint: {
    "raster-opacity": 0.8, // Make overlay more visible
    "raster-fade-duration": 300,
  },
  minzoom: 0,
  maxzoom: 15,
};

export default function MapSection() {
  const { theme } = useTheme();
  const [mapTheme, setMapTheme] = useState("light"); // Start with dark theme like target
  const [tileError, setTileError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const searchParams = useSearchParams();
  const lat = searchParams.get("lat");
  const lon = searchParams.get("lon");

  const defaultLat = lat ? Number(lat) : Number(DEFAULT_LOCATION.coord.lat);
  const defaultLon = lon ? Number(lon) : Number(DEFAULT_LOCATION.coord.lon);

  const [viewport, setViewport] = useState({
    latitude: defaultLat,
    longitude: defaultLon,
    zoom: 6, // Better zoom level for weather visualization
    pitch: 0,
    bearing: 0,
  });

  const [mapCode, setMapCode] = useState("precipitation_new"); // Start with precipitation like target

  useEffect(() => {
    if (theme === "system") {
      const darkModeMediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
      setMapTheme(darkModeMediaQuery.matches ? "dark" : "light");
    } else {
      setMapTheme(theme || "light");
    }
  }, [theme]);

  useEffect(() => {
    setViewport((prev) => ({
      ...prev,
      latitude: defaultLat,
      longitude: defaultLon,
    }));
  }, [defaultLat, defaultLon]);

  // Test different tile URL formats
  const getTileUrl = (code: string) => {
    // Try different URL formats if one doesn't work
    const formats = [`https://tile.openweathermap.org/map/${code}/{z}/{x}/{y}.png?appid=${OPENWEATHERMAP_TOKEN}`, `https://maps.openweathermap.org/maps/2.0/weather/${code}/{z}/{x}/{y}?appid=${OPENWEATHERMAP_TOKEN}`];
    return formats[0]; // Start with first format
  };

  const handleMapCodeChange = (newCode: string) => {
    setIsLoading(true);
    setTileError(null);
    setMapCode(newCode);

    // Reset loading after a short delay
    setTimeout(() => setIsLoading(false), 1000);
  };

  if (!MAPBOX_TOKEN) {
    return <div className="p-4 text-red-500">Mapbox token tidak ditemukan</div>;
  }

  if (!OPENWEATHERMAP_TOKEN) {
    return <div className="p-4 text-red-500">OpenWeatherMap token tidak ditemukan</div>;
  }

  return (
    <Card className="relative overflow-hidden overscroll-contain p-0">
      <div className="absolute right-0 top-0 z-10 m-4">
        <Select value={mapCode} onValueChange={handleMapCodeChange}>
          <SelectTrigger aria-label="Map layer" className="w-[200px] bg-black/50 text-white border-white/20 backdrop-blur-sm">
            <SelectValue placeholder="Pilih Layer" />
          </SelectTrigger>
          <SelectContent align="end" className="bg-black/80 text-white border-white/20">
            <SelectGroup>
              {weatherTiles.map((tile) => (
                <SelectItem key={tile.code} value={tile.code} className="hover:bg-white/10 focus:bg-white/10">
                  {tile.label}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>

      {isLoading && <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-20 bg-black/50 text-white px-4 py-2 rounded">Loading weather data...</div>}

      <ReactMapGL
        {...viewport}
        onMove={(evt) => setViewport(evt.viewState)}
        mapboxAccessToken={MAPBOX_TOKEN}
        mapStyle={`mapbox://styles/mapbox/${mapTheme}-v11`}
        style={{ width: "100%", height: "400px" }}
        onError={(e) => {
          console.error("Map error:", e);
          setTileError(e.error?.toString() || "Map loading error");
        }}
        attributionControl={false} // Clean up attribution for better look
      >
        <Source
          key={`weather-${mapCode}`} // Force re-render on code change
          id="weatherSource"
          type="raster"
          tiles={[getTileUrl(mapCode)]}
          tileSize={256}
          attribution="OpenWeatherMap"
        >
          <Layer {...weatherLayer} />
        </Source>
      </ReactMapGL>

      {tileError && (
        <div className="absolute bottom-4 left-4 bg-red-500/80 text-white p-3 rounded backdrop-blur-sm">
          <div className="font-semibold">Weather Tile Error:</div>
          <div className="text-sm">{tileError}</div>
          <div className="text-xs mt-1">Periksa API key OpenWeatherMap atau coba layer lain</div>
        </div>
      )}

      {/* Debug info - remove in production */}
      <div className="absolute bottom-4 right-4 bg-black/50 text-white p-2 rounded text-xs backdrop-blur-sm">
        <div>Layer: {mapCode}</div>
        <div>Theme: {mapTheme}</div>
        <div>Zoom: {viewport.zoom.toFixed(1)}</div>
      </div>
    </Card>
  );
}
