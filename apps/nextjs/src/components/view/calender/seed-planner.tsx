import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getSeeds, createSeed, getHoltWinterDaily, getLSTMDaily } from "@/lib/fetch/files.fetch";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Combobox } from "@/components/combobox";
import { format } from "date-fns";
import { Label } from "@/components/ui/label";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { 
  Cloud, 
  Calendar, 
  Clock, 
  Sprout, 
  Droplets, 
  Thermometer, 
  Wind,
  Info,
  CheckCircle,
  AlertCircle,
  XCircle
} from "lucide-react";

interface SeedItem {
  name: string;
  duration: number;
}

type ModelType = "holt-winters" | "lstm";

const RAIN_MIN = 5.7;
const RAIN_MAX = 16.7;
const TEMP_MIN = 24;
const TEMP_MAX = 29;
const HUM_MIN = 33;
const HUM_MAX = 90;

const GARAP_DURATION = 5;
const SEMAI_DURATION = 20;

export default function CalendarSeedPlanner() {
  const queryClient = useQueryClient();
  const [selectedSeedName, setSelectedSeedName] = useState<string>("");
  const [duration, setDuration] = useState<number>(0);
  const [startDate, setStartDate] = useState<string>("");
  const [semaiStatus, setSemaiStatus] = useState<string>("belum");
  const [showGrid, setShowGrid] = useState(false);
  const [currentModel, setCurrentModel] = useState<ModelType>("holt-winters");

  const { data: seedsData } = useQuery({
    queryKey: ["get-seeds-planner"],
    queryFn: () => getSeeds(1, 100),
    refetchOnWindowFocus: false,
  });

  // Query data berdasarkan model yang dipilih
  const { data: forecastData, isLoading: isForecastLoading } = useQuery({
    queryKey: ["forecast-seed-planner", currentModel],
    queryFn: async () => {
      if (currentModel === "lstm") {
        const data = await getLSTMDaily(1, 731);
        return data;
      } else {
        const data = await getHoltWinterDaily(1, 731);
        return data;
      }
    },
    refetchOnWindowFocus: false,
  });

  // Listen untuk perubahan model dari monthly calendar
  useEffect(() => {
    const handleModelChange = (event: CustomEvent) => {
      const newModel = event.detail.model as ModelType;
      setCurrentModel(newModel);
      
      // Jika sedang menampilkan grid, refresh data dengan model baru
      if (showGrid) {
        setShowGrid(false);
        setTimeout(() => setShowGrid(true), 100);
      }
    };

    window.addEventListener('modelChanged', handleModelChange as EventListener);

    // Cleanup
    return () => {
      window.removeEventListener('modelChanged', handleModelChange as EventListener);
    };
  }, [showGrid]);

  const createSeedMutation = useMutation({
    mutationFn: createSeed,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["get-seeds-planner"] }),
  });

  const handleSeedChange = (name: string) => {
    setSelectedSeedName(name);
    const match = seedsData?.items.find((s: SeedItem) => s.name.toLowerCase() === name.toLowerCase());
    if (match) setDuration(match.duration);
  };

  const handleSubmit = async () => {
    if (!selectedSeedName || !duration || !startDate) return;

    const exists = seedsData?.items.find((s: SeedItem) => s.name.toLowerCase() === selectedSeedName.toLowerCase() && s.duration === duration);
    if (!exists) {
      await createSeedMutation.mutateAsync({ name: selectedSeedName, duration });
    }
    setShowGrid(true);
  };

  const calculateActualStartDate = () => {
    if (!startDate) return null;
    const targetPlantDate = new Date(startDate);
    const daysToSubtract = semaiStatus === "belum" ? GARAP_DURATION + SEMAI_DURATION : GARAP_DURATION;
    const actualStartDate = new Date(targetPlantDate);
    actualStartDate.setDate(actualStartDate.getDate() - daysToSubtract);
    return actualStartDate;
  };

  const getWeatherColor = (rain: number, temp: number, hum: number) => {
    const isRainExtreme = rain < RAIN_MIN || rain > RAIN_MAX;
    const isTempExtreme = temp < TEMP_MIN || temp > TEMP_MAX;
    const isHumExtreme = hum < HUM_MIN || hum > HUM_MAX;

    const extremeCount = [isRainExtreme, isTempExtreme, isHumExtreme].filter(Boolean).length;

    if (extremeCount === 0) return "bg-emerald-100 border-emerald-200 text-emerald-800"; // semua sesuai
    if (extremeCount >= 2) return "bg-red-100 border-red-200 text-red-800"; // ada 2+ ekstrem → bahaya
    return "bg-amber-100 border-amber-200 text-amber-800"; // ada 1 ekstrem → bisa tanam tapi hati-hati
  };

  const getWeatherIcon = (rain: number, temp: number, hum: number) => {
    const isRainExtreme = rain < RAIN_MIN || rain > RAIN_MAX;
    const isTempExtreme = temp < TEMP_MIN || temp > TEMP_MAX;
    const isHumExtreme = hum < HUM_MIN || hum > HUM_MAX;

    const extremeCount = [isRainExtreme, isTempExtreme, isHumExtreme].filter(Boolean).length;

    if (extremeCount === 0) return <CheckCircle className="w-3 h-3" />;
    if (extremeCount >= 2) return <XCircle className="w-3 h-3" />;
    return <AlertCircle className="w-3 h-3" />;
  };

  const renderGrid = () => {
    if (!selectedSeedName || !duration || !startDate || !forecastData) return null;
    
    const actualStartDate = calculateActualStartDate();
    if (!actualStartDate) return null;

    const preparationDays = semaiStatus === "belum" ? GARAP_DURATION + SEMAI_DURATION : GARAP_DURATION;
    const totalGridDays = preparationDays + duration;

    const gridDays = Array.from({ length: totalGridDays }, (_, i) => {
      const date = new Date(actualStartDate);
      date.setDate(date.getDate() + i);
                              
      let type = "";
      let bgColor = "";
      let icon = null;
      let rain = 0,
        temp = 0,
        hum = 0;

      if (i < preparationDays) {
        if (semaiStatus === "belum") {
          if (i < GARAP_DURATION) {
            type = "Garap";
            bgColor = "bg-orange-50 border-orange-200 text-orange-800";
            icon = <Sprout className="w-3 h-3" />;
          } else {
            type = "Semai";
            bgColor = "bg-blue-50 border-blue-200 text-blue-800";
            icon = <Sprout className="w-3 h-3" />;
          }
        } else {
          type = "Garap";
          bgColor = "bg-orange-50 border-orange-200 text-orange-800";
          icon = <Sprout className="w-3 h-3" />;
        }
      } else {
        type = "Masa Tanam";
        const forecastDay = forecastData.items.find((f: any) => 
          format(new Date(f.forecast_date), "yyyy-MM-dd") === format(date, "yyyy-MM-dd")
        );

        if (forecastDay) {
          rain = forecastDay.parameters?.RR_imputed?.forecast_value ?? 0;
          temp = forecastDay.parameters?.TAVG?.forecast_value ?? 0;
          hum = forecastDay.parameters?.RH_AVG_preprocessed?.forecast_value ?? 0;
          bgColor = getWeatherColor(rain, temp, hum);
          icon = getWeatherIcon(rain, temp, hum);
        } else {
          bgColor = "bg-gray-50 border-gray-200 text-gray-800";
          icon = <Info className="w-3 h-3" />;
        }

        // Panen warna kuning
        const dayInCycle = i - preparationDays;
        if (dayInCycle >= duration - 20) {
          type = "Panen";
          bgColor = "bg-yellow-50 border-yellow-200 text-yellow-800";
          icon = <Calendar className="w-3 h-3" />;
        }
      }

      return {
        day: i + 1,
        date: format(date, "MMM d"),
        type,
        bgColor,
        icon,
        rain,
        temp,
        hum,
      };
    });

    return (
      <Card className="mt-6">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Calendar className="w-5 h-5" />
            Jadwal Penanaman
          </CardTitle>

          {/* Model indicator */}
          <div className="flex items-center gap-3 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
            <Cloud className="w-5 h-5 text-blue-600" />
            <div className="flex flex-col sm:flex-row sm:items-center gap-2">
              <span className="text-sm font-medium text-blue-800">Model Prediksi:</span>
              <Badge 
                variant={currentModel === "lstm" ? "default" : "secondary"}
                className="w-fit"
              >
                {currentModel === "lstm" ? "LSTM Neural Network" : "Holt Winters"}
              </Badge>
              {isForecastLoading && (
                <span className="text-xs text-blue-600 animate-pulse flex items-center gap-1">
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
                  Memuat data...
                </span>
              )}
            </div>
          </div>

          {/* Legend */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 text-sm">
              <div className="w-4 h-4 bg-orange-200 rounded border border-orange-300"></div>
              <span className="text-gray-700">Garap ({GARAP_DURATION} hari)</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-4 h-4 bg-blue-200 rounded border border-blue-300"></div>
              <span className="text-gray-700">Semai ({SEMAI_DURATION} hari)</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-4 h-4 bg-emerald-200 rounded border border-emerald-300"></div>
              <span className="text-gray-700">Masa Tanam</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-4 h-4 bg-yellow-200 rounded border border-yellow-300"></div>
              <span className="text-gray-700">Panen</span>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          <TooltipProvider>
            <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 xl:grid-cols-7 gap-3">
              {gridDays.map((item, idx) => (
                <Tooltip key={idx}>
                  <TooltipTrigger asChild>
                    <div className={`
                      p-3 rounded-lg border-2 text-center transition-all duration-200
                      hover:scale-105 hover:shadow-md cursor-pointer
                      ${item.bgColor}
                    `}>
                      <div className="flex items-center justify-center mb-1">
                        {item.icon}
                      </div>
                      <div className="font-semibold text-sm">{item.date}</div>
                      <div className="text-xs font-medium">{item.type}</div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-xs">
                    <div className="space-y-2">
                      <p className="text-sm font-semibold border-b pb-1">
                        {item.date} - {item.type}
                      </p>
                      {item.type === "Masa Tanam" && (
                        <div className="space-y-1">
                          <div className="flex items-center gap-2 text-xs">
                            <Droplets className="w-3 h-3 text-blue-500" />
                            <span>Curah hujan: {item.rain.toFixed(1)} mm</span>
                          </div>
                          <div className="flex items-center gap-2 text-xs">
                            <Thermometer className="w-3 h-3 text-red-500" />
                            <span>Suhu: {item.temp.toFixed(1)} °C</span>
                          </div>
                          <div className="flex items-center gap-2 text-xs">
                            <Wind className="w-3 h-3 text-green-500" />
                            <span>Kelembaban: {item.hum.toFixed(1)} %</span>
                          </div>
                          <div className="flex items-center gap-2 text-xs text-gray-500 pt-1 border-t">
                            <Cloud className="w-3 h-3" />
                            <span>Model: {currentModel.toUpperCase()}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </TooltipContent>
                </Tooltip>
              ))}
            </div>
          </TooltipProvider>

          <Separator className="my-6" />

          {/* Summary */}
          <Card className="bg-gradient-to-r from-green-50 to-blue-50">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Clock className="w-4 h-4" />
                Ringkasan Jadwal
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 uppercase tracking-wide">Mulai Persiapan</span>
                  <span className="text-sm font-semibold text-gray-800">
                    {format(actualStartDate, "d MMM yyyy")}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 uppercase tracking-wide">Mulai Tanam</span>
                  <span className="text-sm font-semibold text-gray-800">
                    {format(new Date(startDate), "d MMM yyyy")}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 uppercase tracking-wide">Estimasi Panen</span>
                  <span className="text-sm font-semibold text-gray-800">
                    {format(new Date(new Date(startDate).getTime() + (duration - 20) * 24 * 60 * 60 * 1000), "d MMM yyyy")}
                  </span>
                </div>
              </div>
              
              <div className="flex items-center gap-2 pt-2 border-t border-gray-200">
                <Cloud className="w-4 h-4 text-blue-600" />
                <span className="text-sm text-blue-700">
                  Prediksi menggunakan model: 
                  <span className="font-semibold ml-1">
                    {currentModel === "lstm" ? "LSTM Neural Network" : "Holt Winters"}
                  </span>
                </span>
              </div>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sprout className="w-5 h-5 text-green-600" />
            Perencanaan Penanaman Benih
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="space-y-2">
              <Label htmlFor="seed-select" className="flex items-center gap-2 text-sm font-medium">
                <Sprout className="w-4 h-4" />
                Pilih Benih
              </Label>
              <Combobox 
                options={seedsData?.items.map((s: SeedItem) => s.name) || []} 
                value={selectedSeedName} 
                onValueChange={handleSeedChange}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="duration-input" className="flex items-center gap-2 text-sm font-medium">
                <Clock className="w-4 h-4" />
                Durasi (Hari)
              </Label>
              <Input 
                id="duration-input"
                type="number" 
                placeholder="Durasi tanam" 
                value={duration} 
                onChange={(e) => setDuration(Number(e.target.value))}
                className="transition-all duration-200 focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="start-date" className="flex items-center gap-2 text-sm font-medium">
                <Calendar className="w-4 h-4" />
                Tanggal Mulai Tanam
              </Label>
              <Input 
                id="start-date"
                type="date" 
                value={startDate} 
                onChange={(e) => setStartDate(e.target.value)}
                className="transition-all duration-200 focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="semai-status" className="flex items-center gap-2 text-sm font-medium">
                <Info className="w-4 h-4" />
                Status Semai
              </Label>
              <Select value={semaiStatus} onValueChange={setSemaiStatus}>
                <SelectTrigger id="semai-status">
                  <SelectValue placeholder="Status Semai" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="belum">Belum Semai</SelectItem>
                  <SelectItem value="sudah">Sudah Semai</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <Button 
            className="mt-6 w-full sm:w-auto px-8 py-2" 
            size="lg"
            onClick={handleSubmit} 
            disabled={!selectedSeedName || !duration || !startDate || isForecastLoading}
          >
            {isForecastLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                Memuat Data...
              </>
            ) : (
              <>
                <Calendar className="w-4 h-4 mr-2" />
                Buat Jadwal Penanaman
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {showGrid && renderGrid()}
    </div>
  );
}
