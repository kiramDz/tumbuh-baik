import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Cloudy, Rainy, Sunny } from "../../../public/svg/weather";
import type { BMKGDataItem } from "@/types/table-schema";

interface CurrentWeatherProps {
  unit: string;
  bmkgCurrent: BMKGDataItem & {
    nama_gampong?: string; 
    kode_gampong?: string; 
  };
}

const CurrentWeatherCard: React.FC<CurrentWeatherProps> = ({ bmkgCurrent, unit }) => {
  const localDate = new Date(bmkgCurrent.local_datetime.replace(" ", "T"));
  const formattedDate = `Today, ${localDate.toLocaleDateString("id-ID", {
    weekday: "long",
    month: "long",
    day: "numeric",
  })}`;

  const getWeatherIcon = (tcc: number) => {
    if (tcc >= 80) return <Cloudy className="w-32 h-32 text-gray-500 transition-transform hover:scale-105" />;
    if (tcc >= 50) return <Rainy className="w-32 h-32 text-blue-500 transition-transform hover:scale-105" />;
    return <Sunny className="w-32 h-32 text-yellow-500 transition-transform hover:scale-105" />;
  };

  return (
    <Card className="relative h-fit w-full md:h-[28rem]">
      <CardContent className="flex flex-col justify-between h-full py-6">
        {/* Location and Time Section */}
        <div className="flex flex-col items-center w-full">
          <div className="space-y-1 text-center">
            <h2 className="text-xl font-bold tracking-tight">Gampong {bmkgCurrent.nama_gampong}, Aceh Besar</h2>
            <p className="text-xs sm:text-sm font-medium text-muted-foreground">{formattedDate}</p>
          </div>
          <div className="transform transition-transform hover:scale-105 duration-300 mt-4">{getWeatherIcon(bmkgCurrent.tcc)}</div>
        </div>

        {/* Temperature Section */}
        <div className="flex flex-col items-center justify-center my-auto">
          <div className="flex items-end">
            <p className="text-8xl font-bold tracking-tighter">{Math.round(bmkgCurrent.t)}°</p>
            <p className="text-2xl font-bold mb-2 ml-1">{unit === "metric" ? "C" : "F"}</p>
          </div>
          <div>
            <p className="text-xl font-semibold text-muted-foreground">{bmkgCurrent.weather_desc}</p>
          </div>
        </div>

        {/* Weather Details Grid */}
        {/* <div className="grid grid-cols-2 gap-4 w-full items-end justify-end mt-auto">
          <div className="flex items-center">
            <Wind className="w-5 h-5 mr-2 aspect-square text-blue-400" />
            <span className="text-xs sm:text-sm  md:text-md text-nowrap">Wind Speed: {bmkgCurrent.ws}</span>
          </div>
          <div className="flex items-center">
            <Droplets className="w-5 h-5 mr-2 aspect-square text-blue-400" />
            <span className="text-xs sm:text-sm  md:text-md text-nowrap">Humidity: {bmkgCurrent.hu}%</span>
          </div>
          <div className="flex items-center">
            <Droplets className="w-5 h-5 mr-2 aspect-square text-blue-400" />
            <span className="text-xs sm:text-sm  md:text-md text-nowrap">Temperature: {bmkgCurrent.t}%</span>
          </div>
        </div> */}
      </CardContent>
    </Card>
  );
};

export default CurrentWeatherCard;
