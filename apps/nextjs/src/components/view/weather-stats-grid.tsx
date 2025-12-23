"use client";

import React, { useMemo } from "react";
import { Wind, Droplets, Eye, Sun } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";

interface WeatherStatsGridProps {
  latestData: any;
}

export const WeatherStatsGrid = React.memo(({ latestData }: WeatherStatsGridProps) => {
  const weatherStats = useMemo(() => [
    {
      icon: <Wind className="w-5 h-5" />,
      label: "Kecepatan Angin",
      value: `${latestData?.ws || 0} km/h`,
      color: "text-gray-900 dark:text-gray-100",
      bg: "bg-gray-50 dark:bg-gray-900/20",
      iconBg: "bg-gray-100 dark:bg-gray-800/30"
    },
    {
      icon: <Droplets className="w-5 h-5" />,
      label: "Kelembaban",
      value: `${latestData?.hu || 0}%`,
      color: "text-gray-900 dark:text-gray-100",
      bg: "bg-gray-50 dark:bg-gray-900/20",
      iconBg: "bg-gray-100 dark:bg-gray-800/30"
    },
    {
      icon: <Eye className="w-5 h-5" />,
      label: "Visibilitas",
      value: "Baik",
      color: "text-gray-900 dark:text-gray-100",
      bg: "bg-gray-50 dark:bg-gray-900/20",
      iconBg: "bg-gray-100 dark:bg-gray-800/30"
    },
    {
      icon: <Sun className="w-5 h-5" />,
      label: "UV Index",
      value: "Sedang",
      color: "text-gray-900 dark:text-gray-100",
      bg: "bg-gray-50 dark:bg-gray-900/20",
      iconBg: "bg-gray-100 dark:bg-gray-800/30"
    }
  ], [latestData]);

  return (
    <Card className="bg-white/90 dark:bg-gray-800/90 border border-gray-200/30 dark:border-gray-700/30 shadow-lg">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <div className="p-1.5 bg-gray-100 dark:bg-gray-800/30 rounded-lg">
            <TrendingUp className="w-5 h-5 text-gray-900 dark:text-gray-100" />
          </div>
          <span className="bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent dark:from-gray-100 dark:to-gray-300">
            Detail Cuaca
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {weatherStats.map((item, index) => (
            <div
              key={index}
              className={`p-3 rounded-xl border border-gray-200/50 dark:border-gray-700/50 ${item.bg} hover:shadow-md transition-shadow`}
            >
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${item.iconBg}`}>
                  <div className={item.color}>
                    {item.icon}
                  </div>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {item.label}
                  </p>
                  <p className={`text-lg font-semibold ${item.color}`}>
                    {item.value}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
});

WeatherStatsGrid.displayName = 'WeatherStatsGrid';