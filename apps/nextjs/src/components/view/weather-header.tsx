// components/ui/weather-header.tsx
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { BMKGApiData } from "@/types/table-schema";
import { getUniqueGampongData } from "@/lib/bmkg-utils";

interface WeatherHeaderProps {
  bmkgData: BMKGApiData[];
  selectedCode: string;
  onGampongChange: (code: string) => void;
}

export const WeatherHeader: React.FC<WeatherHeaderProps> = ({ bmkgData, selectedCode, onGampongChange }) => {
  const uniqueGampongs = getUniqueGampongData(bmkgData);
  const selected = uniqueGampongs.find((item) => item.kode_gampong === selectedCode);
  const tanggal = selected?.tanggal_data ? new Date(selected.tanggal_data) : new Date();

  const day = tanggal.toLocaleDateString("id-ID", { weekday: "long" });
  const date = tanggal.getDate();
  const month = tanggal.toLocaleDateString("id-ID", { month: "long" });

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-start gap-4 md:gap-10 p-4 rounded-xl">
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

      {/* Gampong Selector */}
      <Select value={selectedCode} onValueChange={onGampongChange}>
        <SelectTrigger className="w-full sm:w-[250px] rounded-md">
          <SelectValue placeholder="Pilih Gampong" />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectLabel>Gampong</SelectLabel>
            {uniqueGampongs.map((item) => (
              <SelectItem key={item.kode_gampong} value={item.kode_gampong}>
                {item.nama_gampong}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
};
