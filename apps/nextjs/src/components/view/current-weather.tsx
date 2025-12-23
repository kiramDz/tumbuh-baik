import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Wind, Droplet } from "lucide-react";
import type { BMKGDataItem } from "@/types/table-schema";

interface CurrentWeatherProps {
  unit: "metric" | "imperial";
  bmkgCurrent: BMKGDataItem;
}

interface WeatherMetricProps {
  icon: React.ComponentType;
  label: string;
  value: string;
}

// Weather metrics component untuk reusability
const WeatherMetric: React.FC<WeatherMetricProps> = ({ icon: Icon, label, value }) => (
  <div className="flex flex-col">
    <div className="flex gap-2 items-center">
      <Icon />
      <span className="text-xs sm:text-sm text-nowrap">{label}</span>
    </div>
    <p className="text-xl sm:text-2xl font-bold">{value}</p>
  </div>
);

interface TemperatureDisplayProps {
  temperature: number;
  unit: "metric" | "imperial";
  description: string;
}

// Main temperature display component
const TemperatureDisplay: React.FC<TemperatureDisplayProps> = ({ temperature, unit, description }) => (
  <div className="flex flex-col gap-2 sm:gap-4 justify-center my-auto">
    <div className="flex items-start">
      <p className="text-6xl sm:text-7xl md:text-8xl font-bold tracking-tighter">{Math.round(temperature)}°</p>
      <p className="text-xl sm:text-2xl font-bold">{unit === "metric" ? "C" : "F"}</p>
    </div>
    <p className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl text-muted-foreground capitalize">{description}</p>
  </div>
);

interface WeatherMetricsProps {
  windSpeed: number;
  humidity: number;
  unit: "metric" | "imperial";
}
// Weather metrics section
const WeatherMetrics: React.FC<WeatherMetricsProps> = ({ windSpeed, humidity, unit }) => (
  <div className="flex gap-4 sm:gap-6">
    <WeatherMetric icon={Wind} label="wind" value={`${Math.round(windSpeed)}${unit === "metric" ? "km/h" : "mph"}`} />
    <WeatherMetric
      icon={Droplet}
      label="humidity" // Fixed: was "wind" before
      value={`${Math.round(humidity)}%`}
    />
  </div>
);

const CurrentWeatherCard: React.FC<CurrentWeatherProps> = ({ bmkgCurrent, unit }) => {
  return (
    <Card className="relative h-fit w-full lg:h-[28rem] border-none shadow-none">
      <CardContent className="flex flex-col justify-center h-full p-4 sm:p-6">
        <TemperatureDisplay temperature={bmkgCurrent.t} unit={unit} description={bmkgCurrent.weather_desc} />
        <WeatherMetrics windSpeed={bmkgCurrent.ws} humidity={bmkgCurrent.hu} unit={unit} />
      </CardContent>
    </Card>
  );
};

export default CurrentWeatherCard;
