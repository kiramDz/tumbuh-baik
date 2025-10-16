"use client";

import React, { useEffect, useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Banner } from "./banner";
import { WeatherHeader } from "./weather-header";
import { WeatherTabs } from "./weather-tabs";
import WeatherIcon from "./weather-icon";
import CurrentWeatherCard from "./current-weather";
import { getBmkgLive } from "@/lib/fetch/files.fetch";
import { getTodayWeather, getHourlyForecastData } from "@/lib/bmkg-utils";
// import { RainbowGlowGradientLineChart } from "./chart/weather-rainbow-chart";
import { WeatherChartTabs } from "./chart/weather-chart-tabs";
interface WeatherDashboardProps {
  unit: "metric" | "imperial";
}

const WeatherDashboard: React.FC<WeatherDashboardProps> = ({ unit }) => {
  const [selectedGampong, setSelectedGampong] = useState<string | null>(null);
  const { data: bmkgApiResponse } = useQuery({
    queryKey: ["bmkg-api"],
    queryFn: getBmkgLive,
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
  const hourlyForecast = useMemo(() => getHourlyForecastData(selectedData), [selectedData]);

  console.log("hourlyForecast:", hourlyForecast);
  console.log("latestData:", latestData);

  return (
    <>
      <div className="bg-inherit min-h-screen flex flex-col space-y-4 md:space-y-6 px-0 ">
        <Banner />
        <WeatherTabs defaultTab="weather">
          {selectedGampong && <WeatherHeader bmkgData={bmkgData} selectedCode={selectedGampong} onGampongChange={setSelectedGampong} />}

          {/* Responsive layout: stack on mobile, side-by-side on desktop */}
          <div className="w-full flex flex-col lg:flex-row lg:justify-between lg:items-start gap-4 lg:gap-6">
            <div className="flex-1 w-full">{latestData && selected && <CurrentWeatherCard bmkgCurrent={{ ...latestData }} unit={unit} />}</div>
            <div className="flex-1 w-full flex justify-center lg:justify-end">{latestData && selected && <WeatherIcon description={latestData.weather_desc} />}</div>
          </div>

          {hourlyForecast.length > 0 ? (
            <>
              <WeatherChartTabs hourlyForecast={hourlyForecast} />
            </>
          ) : (
            <div className="text-center py-8">Loading ...</div>
          )}
        </WeatherTabs>
      </div>
    </>
  );
};

export default WeatherDashboard;
