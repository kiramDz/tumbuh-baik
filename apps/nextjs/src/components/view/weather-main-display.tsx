"use client";

import React from "react";
import { 
  Thermometer, 
  Droplets, 
  Wind
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import WeatherIcon from "./weather-icon";

interface WeatherMainDisplayProps {
  latestData: any;
  unit: "metric" | "imperial";
}

export const WeatherMainDisplay = React.memo(({ 
  latestData, 
  unit 
}: WeatherMainDisplayProps) => {
  
  const temperature = unit === "metric" ? latestData.t : (latestData.t * 9/5) + 32;
  const tempUnit = unit === "metric" ? "°C" : "°F";

  return (
    <div className="grid grid-cols-1 gap-6">
      
      {/* Current Weather Card - Full Width */}
      <Card className="bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white border-0 shadow-2xl overflow-hidden relative">
        
        {/* Decorative Elements */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -mr-32 -mt-32 blur-3xl" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/5 rounded-full -ml-24 -mb-24 blur-2xl" />
        
        <CardHeader className="relative pb-2">
          <div className="flex items-start justify-between">
            <div>
              <CardTitle className="text-white/90 text-sm font-medium mb-1">
                Cuaca Saat Ini
              </CardTitle>
              <div className="text-7xl font-bold mb-2 drop-shadow-lg">
                {Math.round(temperature)}
                <span className="text-4xl font-normal text-white/80">{tempUnit}</span>
              </div>
              <p className="text-xl text-white/90 font-medium">
                {latestData.weather_desc || 'Tidak ada data'}
              </p>
            </div>
            
            {/* Weather Icon */}
            <div className="relative">
              <div className="absolute inset-0 bg-white/10 rounded-full blur-xl" />
              <div className="relative w-24 h-24">
                <WeatherIcon description={latestData?.weather_desc || ''} />
              </div>
            </div>
          </div>
        </CardHeader>

        <CardContent className="relative pt-4">
          {/* Separator */}
          <Separator className="bg-white/20 mb-4" />

          {/* Weather Details Grid */}
          <div className="grid grid-cols-2 gap-4">
            
            {/* Feels Like */}
            <div className="flex items-center gap-3 bg-white/10 backdrop-blur-sm rounded-lg p-3 border border-white/20">
              <div className="p-2 bg-white/20 rounded-lg">
                <Thermometer className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-xs text-white/70">Terasa Seperti</p>
                <p className="text-lg font-semibold text-white">
                  {Math.round(temperature - 2)}{tempUnit}
                </p>
              </div>
            </div>

            {/* Humidity */}
            <div className="flex items-center gap-3 bg-white/10 backdrop-blur-sm rounded-lg p-3 border border-white/20">
              <div className="p-2 bg-white/20 rounded-lg">
                <Droplets className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-xs text-white/70">Kelembaban</p>
                <p className="text-lg font-semibold text-white">
                  {latestData.hu || 0}%
                </p>
              </div>
            </div>

            {/* Wind Speed */}
            <div className="flex items-center gap-3 bg-white/10 backdrop-blur-sm rounded-lg p-3 border border-white/20">
              <div className="p-2 bg-white/20 rounded-lg">
                <Wind className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-xs text-white/70">Kec. Angin</p>
                <p className="text-lg font-semibold text-white">
                  {latestData.ws || 0} km/h
                </p>
              </div>
            </div>

            {/* Wind Direction */}
            <div className="flex items-center gap-3 bg-white/10 backdrop-blur-sm rounded-lg p-3 border border-white/20">
              <div className="p-2 bg-white/20 rounded-lg">
                <Wind className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-xs text-white/70">Arah Angin</p>
                <p className="text-lg font-semibold text-white">
                  {latestData.wd_deg || 0}°
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

    </div>
  );
});

WeatherMainDisplay.displayName = 'WeatherMainDisplay';