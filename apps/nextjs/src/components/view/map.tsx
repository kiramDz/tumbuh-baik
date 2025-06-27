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

const weatherLayer: LayerProps = {
  id: "weatherLayer",
  type: "raster",
  minzoom: 1,
  maxzoom: 10,
};

export default function MapSection() {
  const { theme } = useTheme();
  const [mapTheme, setMapTheme] = useState("light");
  const [tileError, setTileError] = useState<string | null>(null);

  const searchParams = useSearchParams();
  const lat = searchParams.get("lat");
  const lon = searchParams.get("lon");

  const defaultLat = lat ? Number(lat) : Number(DEFAULT_LOCATION.coord.lat);
  const defaultLon = lon ? Number(lon) : Number(DEFAULT_LOCATION.coord.lon);

  const [viewport, setViewport] = useState({
    latitude: defaultLat,
    longitude: defaultLon,
    zoom: 5, // Zoom awal lebih rendah
    pitch: 0, // Reset pitch untuk testing
    bearing: 0,
  });

  const [mapCode, setMapCode] = useState("clouds_new");

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

  if (!MAPBOX_TOKEN) {
    return <div className="p-4 text-red-500">Mapbox token tidak ditemukan</div>;
  }

  if (!OPENWEATHERMAP_TOKEN) {
    return <div className="p-4 text-red-500">OpenWeatherMap token tidak ditemukan</div>;
  }

  return (
    <Card className="relative overflow-hidden overscroll-contain p-0 md:p-0">
      <div className="absolute right-0 z-10 m-2">
        <Select value={mapCode} onValueChange={setMapCode}>
          <SelectTrigger aria-label="Map layer" className="w-[200px]">
            <SelectValue placeholder="Pilih Layer" />
          </SelectTrigger>
          <SelectContent align="end">
            <SelectGroup>
              {weatherTiles.map((tile) => (
                <SelectItem key={tile.code} value={tile.code}>
                  {tile.label}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>

      <ReactMapGL
        {...viewport}
        onMove={(evt) => setViewport(evt.viewState)}
        mapboxAccessToken={MAPBOX_TOKEN}
        mapStyle={`mapbox://styles/mapbox/${mapTheme}-v11`}
        style={{ width: "100%", height: "500px" }}
        onError={(e) => setTileError(e.error.toString())}
      >
        <Source key={mapCode} id="weatherSource" type="raster" tiles={[`https://tile.openweathermap.org/map/${mapCode}/{z}/{x}/{y}.png?appid=${OPENWEATHERMAP_TOKEN}`]} tileSize={256}>
          <Layer {...weatherLayer} />
        </Source>
      </ReactMapGL>

      {tileError && <div className="absolute bottom-0 left-0 bg-red-500 text-white p-2">Error: {tileError}</div>}
    </Card>
  );
}
