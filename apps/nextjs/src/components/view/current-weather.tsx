import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Wind, Droplet } from "lucide-react";
import { Cloudy, Rainy, Sunny } from "../../../public/svg/weather";
import type { BMKGDataItem } from "@/types/table-schema";

interface CurrentWeatherProps {
  unit: string;
  bmkgCurrent: BMKGDataItem;
}

const CurrentWeatherCard: React.FC<CurrentWeatherProps> = ({ bmkgCurrent, unit }) => {
  return (
    <Card className="relative h-fit w-full md:h-[28rem] border-none shadow-none">
      <CardContent className="flex flex-col justify-between h-full py-6">
        {/* Temperature Section */}
        <div className="flex flex-col justify-center my-auto">
          <div className="flex items-start">
            <p className="text-8xl font-bold tracking-tighter">{Math.round(bmkgCurrent.t)}°</p>
            <p className="text-2xl font-bold mb-2 ml-1">{unit === "metric" ? "C" : "F"}</p>
          </div>
          <div>
            <p className="text-xl font-semibold text-muted-foreground">{bmkgCurrent.weather_desc}</p>
          </div>
        </div>
        <div className="flex gap-3">
          <div className="flex flex-col">
            <div>
              <Wind />
              <p className="text-sm">wind</p>
            </div>
            <p className="text-2xl font-bold mb-2 ml-1">24 km/h</p>
          </div>
          <div className="flex flex-col">
            <div>
              <Droplet />
              <p className="text-sm">Humidty</p>
            </div>
            <p className="text-2xl font-bold mb-2 ml-1">79 %</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default CurrentWeatherCard;
