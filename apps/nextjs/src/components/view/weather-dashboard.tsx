"use client";

import React, { useEffect, useState, useMemo } from "react";
import { WeatherData } from "@/types/weather";
import DayDuration from "./day-duration";
import WeatherConclusion from "./weather-conclusion";
import TemperatureHumidityChart from "./temp-humidity";
import CurrentWeatherCard from "./current-weather";
import WeatherDashboardSkeleton from "../dashboard-skeleton";
import WindPressureCard from "./wind-pressure";
import HourlyForecast from "./hourly-forecast";
import { Banner } from "./banner";
import { WeatherHeader } from "./weather-header";
import { WeatherTabs } from "./weather-tabs";
import { getBmkgApi, getBmkgSummary } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";
import { getTodayWeather, getChartData, getHourlyForecastData, getWeatherConclusionFromDailyData, getTodayWeatherConlusion, getFinalConclusion } from "@/lib/bmkg-utils";
import type { PlantSummaryData } from "@/types/table-schema";

interface WeatherDashboardProps {
  weatherData: WeatherData;
  unit: "metric" | "imperial";
}

const WeatherDashboard: React.FC<WeatherDashboardProps> = ({ weatherData, unit }) => {
  const { currentWeather } = weatherData;
  const [selectedGampong, setSelectedGampong] = useState<string | null>(null);
  const now = new Date();
  const end = new Date();

  const { data: bmkgApiResponse } = useQuery({
    queryKey: ["bmkg-api"],
    queryFn: getBmkgApi,
  });

  const { data: bmkgSummary } = useQuery({
    queryKey: ["bmkg-summary"],
    queryFn: getBmkgSummary,
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

  const chartData = useMemo(() => getChartData(selectedData), [selectedData]);

  const latestData = useMemo(() => getTodayWeather(selectedData), [selectedData]);

  const forecastData = useMemo(() => getHourlyForecastData(selectedData), [selectedData]);

  const todayOnlyData = useMemo(() => getTodayWeatherConlusion(selectedData), [selectedData]);

  const conclusion = useMemo(() => getWeatherConclusionFromDailyData(todayOnlyData), [todayOnlyData]);

  const currentMonth = useMemo(() => {
    const today = new Date();
    return `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}`;
  }, []);

  const currentSummary = useMemo(() => {
    return bmkgSummary?.find((item: PlantSummaryData) => item.month === currentMonth);
  }, [bmkgSummary, currentMonth]);

  const finalConclusion = useMemo(() => {
    return getFinalConclusion(conclusion, currentSummary?.status ?? "tidak cocok tanam");
  }, [conclusion, currentSummary?.status]);

  if (!bmkgApiResponse || !bmkgSummary)
    return (
      <div>
        <WeatherDashboardSkeleton />
      </div>
    );

  end.setDate(now.getDate() + 1);
  end.setHours(23, 59, 59);

  return (
    <div className="bg-inherit min-h-screen flex flex-col">
      <Banner />
      <WeatherTabs defaultTab="weather">
        {selectedGampong && <WeatherHeader bmkgData={bmkgData} selectedCode={selectedGampong} onGampongChange={setSelectedGampong} />}
        <div className=" grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 py-4">
          {latestData && selected && <CurrentWeatherCard bmkgCurrent={{ ...latestData, nama_gampong: selected.nama_gampong }} unit={unit} />}

          <div className="grid grid-rows-2 gap-4">
            {latestData && <WindPressureCard bmkgCurrent={latestData} unit={unit} />}

            <HourlyForecast forecast={forecastData} unit={unit} />
          </div>
          <WeatherConclusion conclusion={finalConclusion} tcc={latestData?.t ?? 0} />

          {chartData.length === 0 ? (
            <p className="text-sm text-muted-foreground">Tidak ada data suhu/kelembapan untuk gampong ini dalam rentang waktu tersebut.</p>
          ) : (
            <TemperatureHumidityChart
              data={{
                list: chartData.map((item) => ({
                  dt: parseInt(item.time) / 1000,
                  main: {
                    temp: item.temperature,
                    humidity: item.humidity,
                  },
                })),
              }}
              unit="metric"
            />
          )}

          <DayDuration data={currentWeather} />
        </div>
      </WeatherTabs>
    </div>
  );
};

export default WeatherDashboard;
