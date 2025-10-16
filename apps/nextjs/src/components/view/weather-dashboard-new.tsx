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
      <div className="bg-inherit min-h-screen flex flex-col space-y-6">
        <Banner />
        <WeatherTabs defaultTab="weather">
          {selectedGampong && <WeatherHeader bmkgData={bmkgData} selectedCode={selectedGampong} onGampongChange={setSelectedGampong} />}
          <div className="w-full flex justify-between mx-auto">
            <div className="flex-1">{latestData && selected && <CurrentWeatherCard bmkgCurrent={{ ...latestData }} unit={unit} />}</div>
            <div className="flex-1 flex justify-end">{latestData && selected && <WeatherIcon description={latestData.weather_desc} />}</div>
          </div>

          {hourlyForecast.length > 0 ? (
            <>
              {/* <RainbowGlowGradientLineChart hourlyForecast={hourlyForecast} /> */}
              <WeatherChartTabs hourlyForecast={hourlyForecast} />
            </>
          ) : (
            <div>Loading ...</div>
          )}
        </WeatherTabs>
      </div>
    </>
  );
};

export default WeatherDashboard;
