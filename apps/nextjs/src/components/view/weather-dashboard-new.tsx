"use client";

import React, { useEffect, useState, useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import dynamic from "next/dynamic";
import { Banner } from "./banner";
import { WeatherHeader } from "./weather-header";
import { WeatherTabs } from "./weather-tabs";
import { WeatherLoading } from "./weather-loading";
import { WeatherError } from "./weather-error";
import { WeatherMainDisplay } from "./weather-main-display";
import { getBmkgLive } from "@/lib/fetch/files.fetch";
import { getTodayWeather, getDailyForecastData, getHourlyForecastData } from "@/lib/bmkg-utils";

// Only lazy load non-critical chart component
const WeatherChart = dynamic(() => import("./weather-chart").then(mod => ({ default: mod.WeatherChart })), {
  loading: () => (
    <div className="min-h-[300px] bg-gray-100 dark:bg-gray-800 rounded-lg" />
  ),
  ssr: false
});

interface WeatherDashboardProps {
  unit: "metric" | "imperial";
}

const WeatherDashboard: React.FC<WeatherDashboardProps> = React.memo(({ unit }) => {
  const [selectedGampong, setSelectedGampong] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const { data: bmkgApiResponse, isLoading, error, refetch } = useQuery({
    queryKey: ["bmkg-api"],
    queryFn: getBmkgLive,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    staleTime: 15 * 60 * 1000,
    gcTime: 20 * 60 * 1000,
  });

  const bmkgData = bmkgApiResponse?.data;

  useEffect(() => {
    if (!selectedGampong && bmkgData?.length) {
      setSelectedGampong(bmkgData[0].kode_gampong);
    }
  }, [bmkgData, selectedGampong]);

  const selected = useMemo(() => {
    return bmkgData?.find((item: any) => item.kode_gampong === selectedGampong) ?? null;
  }, [bmkgData, selectedGampong]);

  const selectedData = useMemo(() => selected?.data ?? [], [selected]);

  const latestData = useMemo(() => getTodayWeather(selectedData), [selectedData]);
  const dailyForecast = useMemo(() => getDailyForecastData(bmkgData), [bmkgData]);
  const hourlyForecast = useMemo(() => getHourlyForecastData(selectedData), [selectedData]);

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await refetch();
    setIsRefreshing(false);
  }, [refetch]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-900 dark:via-blue-900/30 dark:to-indigo-900/30">
      <div className="flex flex-col space-y-6 p-6">
        
        {/* Banner Component */}
        <Banner />

        {/* Weather Tabs Wrapper */}
        <WeatherTabs defaultTab="weather">
          {isLoading ? (
            <WeatherLoading />
          ) : error ? (
            <WeatherError error={error} onRetry={handleRefresh} />
          ) : (
            <div className="space-y-6">
              
              {/* Weather Header */}
              {selectedGampong && (
                <WeatherHeader 
                  bmkgData={bmkgData} 
                  selectedCode={selectedGampong}
                  onGampongChange={setSelectedGampong}
                />
              )}

              {/* Main Weather Display */}
              {latestData && selected && (
                <WeatherMainDisplay 
                  latestData={latestData}
                  unit={unit}
                />
              )}

              {/* Weather Chart */}
              <WeatherChart 
                hourlyForecast={hourlyForecast} 
              />
            </div>
          )}
        </WeatherTabs>
      </div>
    </div>
  );
});

WeatherDashboard.displayName = 'WeatherDashboard';

export default WeatherDashboard;