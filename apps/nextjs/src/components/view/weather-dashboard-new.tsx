"use client";

import React, { useEffect, useState, useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { Banner } from "./banner";
import { WeatherHeader } from "./weather-header";
import { WeatherTabs } from "./weather-tabs";
import { WeatherMainDisplay } from "./weather-main-display";
import { WeatherChart } from "./weather-chart";
import { WeatherLoading } from "./weather-loading";
import { WeatherError } from "./weather-error";
import { getBmkgLive } from "@/lib/fetch/files.fetch";
import { getTodayWeather, getDailyForecastData, getHourlyForecastData } from "@/lib/bmkg-utils";

interface WeatherDashboardProps {
  unit: "metric" | "imperial";
}

const WeatherDashboard: React.FC<WeatherDashboardProps> = ({ unit }) => {
  const [selectedGampong, setSelectedGampong] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const { data: bmkgApiResponse, isLoading, error, refetch } = useQuery({
    queryKey: ["bmkg-api"],
    queryFn: getBmkgLive,
    refetchOnWindowFocus: false,
    staleTime: 10 * 60 * 1000,
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

  const selectedData = useMemo(() => {
    return selected?.data ?? [];
  }, [selected]);

  const latestData = useMemo(() => getTodayWeather(selectedData), [selectedData]);
  const dailyForecast = useMemo(() => getDailyForecastData(bmkgData), [bmkgData]);
  const hourlyForecast = useMemo(() => getHourlyForecastData(selectedData), [selectedData]);

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await refetch();
    setTimeout(() => setIsRefreshing(false), 500);
  }, [refetch]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-900 dark:via-blue-900/30 dark:to-indigo-900/30">
      
      {/* Background decoration */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none opacity-30">
        <div className="absolute top-20 left-20 w-64 h-64 bg-blue-200/20 rounded-full blur-3xl" />
        <div className="absolute bottom-32 right-32 w-48 h-48 bg-indigo-200/20 rounded-full blur-2xl" />
      </div>

      <div className="relative z-10 flex flex-col space-y-6 p-6">
        
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
};

export default WeatherDashboard;