// components/ui/weather-header.tsx
import type { BMKGApiData } from "@/types/table-schema";

interface WeatherHeaderProps {
  bmkgData: BMKGApiData[];
}

export const WeatherHeader: React.FC<WeatherHeaderProps> = ({ bmkgData }) => {
  const tanggal = bmkgData?.[0]?.tanggal_data ? new Date(bmkgData[0].tanggal_data) : new Date();

  const day = tanggal.toLocaleDateString("id-ID", { weekday: "long" });
  const date = tanggal.getDate();
  const month = tanggal.toLocaleDateString("id-ID", { month: "long" });

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 md:gap-10 p-4 rounded-xl">
      {/* Date Display */}
      <div className="flex items-center gap-4">
        <div className="flex items-center justify-center w-16 h-16 border rounded-xl">
          <span className="text-3xl font-semibold text-teal-800">{date}</span>
        </div>
        <div className="flex flex-col">
          <span className="text-gray-500">{day}</span>
          <span className="text-gray-700 font-medium">{month}</span>
        </div>
      </div>
    </div>
  );
};
