"use client";

import React, { useEffect, useState } from "react";
import { WeatherData } from "@/types/weather";
import DayDuration from "./day-duration";
import AirPollutionChart from "./air-pollution";
import TemperatureHumidityChart from "./temp-humidity";
// import ClientMap from "@/components/views/client-map";
import CurrentWeatherCard from "./current-weather";
import WindPressureCard from "./wind-pressure";
import HourlyForecast from "./hourly-forecast";
import { Banner } from "./banner";
import { WeatherTabs } from "./weather-tabs";
import { getBmkgApi } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";

interface ChartDataPoint {
  time: string;
  temperature: number;
  humidity: number;
}

interface WeatherDashboardProps {
  weatherData: WeatherData;
  unit: "metric" | "imperial";
}

const WeatherDashboard: React.FC<WeatherDashboardProps> = ({ weatherData, unit }) => {
  const { currentWeather, forecast, airPollution } = weatherData;
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);

  // Fetch BMKG data
  const {
    data: bmkgApiResponse,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["bmkg-api"],
    queryFn: getBmkgApi,
  });

  const bmkgData = bmkgApiResponse?.data;
  const selected = bmkgData?.find((item) => item.kode_gampong === "11.06.02.2001");
  const latestData = selected?.data?.[selected.data.length - 1];

  useEffect(() => {
    if (!selected || !selected.data) return;

    const today = new Date("2025-05-06T14:00:00");
    const end = new Date("2025-05-08T23:00:00");

    const filtered = selected.data.filter((item: any) => {
      const dt = new Date(item.local_datetime.replace(" ", "T"));
      return dt >= today && dt <= end;
    });

    const mapped = filtered.map((item: any) => ({
      time: new Date(item.local_datetime.replace(" ", "T")).getTime().toString(),
      temperature: item.t,
      humidity: item.hu,
    }));

    setChartData(mapped);
  }, [selected]);

  if (isLoading) return <div>Loading...</div>;

  if (error || !bmkgData) return <div>Error loading BMKG data.</div>;

  const hourlyForecastData = forecast.list.slice(0, 5).map((item) => ({
    time: new Date(item.dt * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    }),
    temperature: Math.round(item.main.temp),
    weather: item.weather[0].main,
  }));

  return (
    <div className="bg-inherit min-h-screen flex flex-col">
      <Banner />
      <WeatherTabs defaultTab="weather">
        <div className="container grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
          {latestData && selected && <CurrentWeatherCard bmkgCurrent={{ ...latestData, nama_gampong: selected.nama_gampong }} unit={unit} />}

          <div className="grid grid-rows-2 gap-4">
            <WindPressureCard currentWeather={currentWeather} unit={unit} />
            <HourlyForecast forecast={hourlyForecastData} unit={unit} />
          </div>
          <AirPollutionChart data={airPollution} />
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
