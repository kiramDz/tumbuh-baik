"use client";

import React, { useEffect, useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { Banner } from "./banner";
import { WeatherHeader } from "./weather-header";
import { WeatherTabs } from "./weather-tabs";
import WeatherIcon from "./weather-icon";
import CurrentWeatherCard from "./current-weather";
import WeatherForecast from "./weather-forecast";
import { ChartLineInteractive } from "../chart-tes";

import { WeatherData } from "@/types/weather";
import { getBmkgApi } from "@/lib/fetch/files.fetch";
import { getTodayWeather } from "@/lib/bmkg-utils";

interface WeatherDashboardProps {
  weatherData: WeatherData;
  unit: "metric" | "imperial";
}

const WeatherDashboard: React.FC<WeatherDashboardProps> = ({ unit, weatherData }) => {
  const [selectedGampong, setSelectedGampong] = useState<string | null>(null);
  const { data: bmkgApiResponse } = useQuery({
    queryKey: ["bmkg-api"],
    queryFn: getBmkgApi,
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

  const selectedData = selected?.data ?? [];
  const latestData = useMemo(() => getTodayWeather(selectedData), [selectedData]);
  return (
    <>
      <div className="bg-inherit min-h-screen flex flex-col space-y-6">
        <Banner />
        <WeatherTabs defaultTab="weather">
          {selectedGampong && <WeatherHeader bmkgData={bmkgData} selectedCode={selectedGampong} onGampongChange={setSelectedGampong} />}
          <div className="w-full flex justify-between mx-auto">
            <div className="flex-1">{latestData && selected && <CurrentWeatherCard bmkgCurrent={{ ...latestData }} unit={unit} />}</div>
            <div className="flex-1 flex justify-center">
              <WeatherIcon />
            </div>
            <div className="flex-1 flex justify-end">
              <WeatherForecast />
            </div>
          </div>
          <ChartLineInteractive />
        </WeatherTabs>
      </div>
    </>
  );
};

export default WeatherDashboard;
