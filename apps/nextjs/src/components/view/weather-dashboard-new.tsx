"use client";

import React, { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Banner } from "./banner";
import { WeatherHeader } from "./weather-header";
import { WeatherTabs } from "./weather-tabs";
import WeatherIcon from "./weather-icon";
import CurrentWeatherCard from "./current-weather";
import { getBmkgLive } from "@/lib/fetch/files.fetch";
import { getTodayWeather, getHourlyForecastData } from "@/lib/bmkg-utils";
import { WeatherChartTabs } from "./chart/weather-chart-tabs";

interface WeatherDashboardProps {
  unit: "metric" | "imperial";
}

const WeatherDashboard: React.FC<WeatherDashboardProps> = ({ unit }) => {
  const { data: bmkgApiResponse } = useQuery({
    queryKey: ["bmkg-api"],
    queryFn: getBmkgLive,
  });
  const bmkgData = bmkgApiResponse?.data;

  const mainData = useMemo(() => {
    return bmkgData?.[0]?.data ?? [];
  }, [bmkgData]);

  const latestData = useMemo(() => getTodayWeather(mainData), [mainData]);

  const hourlyForecast = useMemo(() => getHourlyForecastData(mainData), [mainData]);

  return (
    <>
      <div className="bg-inherit min-h-screen flex flex-col space-y-4 md:space-y-6 px-0 ">
        <Banner />
        <WeatherTabs defaultTab="weather">
          <WeatherHeader bmkgData={bmkgData} />

          <div className="w-full flex flex-col lg:flex-row lg:justify-between lg:items-start gap-4 lg:gap-6">
            <div className="flex-1 w-full">{latestData && <CurrentWeatherCard bmkgCurrent={{ ...latestData }} unit={unit} />}</div>
            <div className="flex-1 w-full flex justify-center lg:justify-end">{latestData && <WeatherIcon description={latestData.weather_desc} />}</div>
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
