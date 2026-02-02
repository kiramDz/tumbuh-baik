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
      color: "text-teal-900 dark:text-teal-100",
      bg: "bg-teal-50 dark:bg-teal-900/20",
      iconBg: "bg-teal-100 dark:bg-teal-800/30"
    },
    {
      icon: <Droplets className="w-5 h-5" />,
      label: "Kelembaban",
      value: `${latestData?.hu || 0}%`,
      color: "text-emerald-900 dark:text-emerald-100",
      bg: "bg-emerald-50 dark:bg-emerald-900/20",
      iconBg: "bg-emerald-100 dark:bg-emerald-800/30"
    },
    {
      icon: <Eye className="w-5 h-5" />,
      label: "Visibilitas",
      value: "Baik",
      color: "text-green-900 dark:text-green-100",
      bg: "bg-green-50 dark:bg-green-900/20",
      iconBg: "bg-green-100 dark:bg-green-800/30"
    },
    {
      icon: <Sun className="w-5 h-5" />,
      label: "UV Index",
      value: "Sedang",
      color: "text-teal-900 dark:text-teal-100",
      bg: "bg-teal-50 dark:bg-teal-900/20",
      iconBg: "bg-teal-100 dark:bg-teal-800/30"
    }
  ], [latestData]);

  return (
    <Card className="bg-white dark:bg-gray-900 border shadow-sm">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <div className="p-1.5 bg-gradient-to-br from-teal-500 to-emerald-600 rounded-lg">
            <TrendingUp className="w-5 h-5 text-white" />
          </div>
          <span className="bg-gradient-to-r from-teal-600 to-emerald-600 bg-clip-text text-transparent">
            Detail Cuaca
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {weatherStats.map((item, index) => (
            <div
              key={index}
              className={`group p-4 rounded-xl border ${item.bg} hover:shadow-lg hover:scale-105 transition-all duration-200 cursor-default`}
            >
              <div className="flex flex-col gap-3">
                <div className={`p-3 rounded-lg ${item.iconBg} w-fit`}>
                  <div className={item.color}>
                    {item.icon}
                  </div>
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1">
                    {item.label}
                  </p>
                  <p className={`text-2xl font-bold ${item.color}`}>
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