import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Wind, Compass, Gauge, Waves, Mountain } from "lucide-react";
import type { BMKGDataItem } from "@/types/table-schema";
import { getWindCondition } from "@/lib/bmkg-utils";

interface CurrentWeatherProps {
  unit: string;
  bmkgCurrent: BMKGDataItem & {
    nama_gampong?: string;
    kode_gampong?: string;
  };
}

const WindPressureCard: React.FC<CurrentWeatherProps> = ({ bmkgCurrent, unit }) => {
  const conditionSummary = getWindCondition(bmkgCurrent.ws, bmkgCurrent.wd);
  return (
    <Card className="w-full mx-auto">
      <CardContent className="p-6">
        <h2 className="text-md font-semibold mb-4 flex items-center">
          <Wind className="w-6 h-6 mr-2" /> Wind & Pressure
        </h2>
        <div className="grid grid-cols-2 gap-0 md:gap-6">
          <div className="text-nowrap">
            <h3 className="text-sm sm:text-md font-semibold mb-2">Wind</h3>
            <div className="space-y-2">
              <p className="flex items-center">
                <Wind className="w-5 h-5 mr-2 aspect-square text-blue-400 dark:text-blue-300" />
                <span className="text-xs sm:text-sm  md:text-md text-nowrap">
                  {Math.round(bmkgCurrent.ws)} {unit === "metric" ? "km/h" : "mph"}
                </span>
              </p>
            </div>
          </div>
          <div className="text-nowrap">
            <h3 className="text-sm sm:text-md font-semibold mb-2">Pressure</h3>
            <div className="space-y-2">
              <p className="flex items-center">
                <Gauge className="w-5 h-5 mr-2 aspect-square text-red-400 dark:text-red-300" />
                <span className="text-xs sm:text-sm  md:text-md text-nowrap">{bmkgCurrent.wd} hPa</span>
              </p>
            </div>
          </div>
        </div>
        {/* kesimpulan */}
        <div className="md:mt-4">
          <p className="text-sm text-muted-foreground italic">{conditionSummary}</p>
        </div>
      </CardContent>
    </Card>
  );
};

export default WindPressureCard;
