import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Clock, CloudSnow } from "lucide-react";
import { ScrollArea, ScrollBar } from "../ui/scroll-area";
import { ClearSky, Cloudy, Rainy, Sunny } from "../../../public/svg/weather";

interface HourlyForecast {
  time: string;
  temperature: number;
  weather: string;
}

interface HourlyForecastProps {
  forecast: HourlyForecast[];
  unit: "metric" | "imperial";
}

// ini icon payah diganti case denagn weather_desc, klo g icon yg muncl to clear
const HourlyForecast: React.FC<HourlyForecastProps> = ({ forecast }) => {
  const getWeatherIcon = (weather: string) => {
    switch (weather.toLowerCase()) {
      case "clear":
        return <Sunny className="w-6 h-6 mt-2" />;
      case "clouds":
        return <Cloudy className="w-6 h-6 mt-2" />;
      case "rain":
        return <Rainy className="w-6 h-6 mt-2" />;
      case "snow":
        return <CloudSnow className="w-6 h-6 mt-2" />;
      default:
        return <ClearSky className="w-6 h-6 mt-2" />;
    }
  };

  return (
    <Card className="w-full h-full overflow-x-scroll">
      <CardContent className="p-4 flex flex-col gap-6">
        <h2 className="text-md font-semibold flex items-center">
          <Clock className="w-5 h-5 mr-2" /> Hourly Forecast
        </h2>
        <ScrollArea className="w-full whitespace-nowrap ">
          <div className="flex items-center justify-evenly w-full h-full gap-4 md:gap-10">
            {forecast.map((hour, index) => (
              <div key={index} className="text-center flex flex-col items-center justify-between">
                <p className="text-xs sm:text-sm font-medium">{hour.time}</p>
                {getWeatherIcon(hour.weather)}
                <p className=" text-xs sm:text-sm">
                  {Math.round(hour.temperature)}°{/* {unit === "metric" ? "C" : "F"} */}
                </p>
              </div>
            ))}
          </div>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default HourlyForecast;
