"use client";

import React from "react";
import { Cloud, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RainbowGlowGradientLineChart } from "./chart/weather-rainbow-chart";

interface WeatherChartProps {
  hourlyForecast: any[];
}

export const WeatherChart = React.memo(({ hourlyForecast }: WeatherChartProps) => {
  return (
    <Card className="bg-white/90 dark:bg-gray-800/90 border border-gray-200/30 dark:border-gray-700/30 shadow-lg">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <div className="p-1.5 bg-gray-100 dark:bg-gray-800/30 rounded-lg">
            <TrendingUp className="w-5 h-5 text-gray-900 dark:text-gray-100" />
          </div>
          <span className="bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent dark:from-gray-100 dark:to-gray-300">
            Grafik Cuaca Per Jam
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        {hourlyForecast.length > 0 ? (
          <RainbowGlowGradientLineChart hourlyForecast={hourlyForecast} />
        ) : (
          <div className="text-center py-8">
            <div className="p-4 bg-gray-100 dark:bg-gray-800/30 rounded-full w-fit mx-auto mb-3">
              <Cloud className="w-12 h-12 text-gray-900 dark:text-gray-100" />
            </div>
            <p className="text-gray-700 dark:text-gray-300 font-medium">
              Data prakiraan per jam tidak tersedia
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
});

WeatherChart.displayName = 'WeatherChart';