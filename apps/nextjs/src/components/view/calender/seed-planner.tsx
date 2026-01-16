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
  XCircle,
  AlertTriangle
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

  useEffect(() => {
    const handleModelChange = (event: CustomEvent) => {
      const newModel = event.detail.model as ModelType;
      setCurrentModel(newModel);
      
      if (showGrid) {
        setShowGrid(false);
        setTimeout(() => setShowGrid(true), 100);
      }
    };

    window.addEventListener('modelChanged', handleModelChange as EventListener);

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

  const getWeatherIssues = (rain: number, temp: number, hum: number) => {
    const issues = [];
    if (rain < RAIN_MIN) issues.push({ type: 'warning', text: `Curah hujan rendah: ${rain.toFixed(1)}mm (min: ${RAIN_MIN}mm)` });
    if (rain > RAIN_MAX) issues.push({ type: 'danger', text: `Curah hujan tinggi: ${rain.toFixed(1)}mm (max: ${RAIN_MAX}mm)` });
    if (temp < TEMP_MIN) issues.push({ type: 'warning', text: `Suhu rendah: ${temp.toFixed(1)}°C (min: ${TEMP_MIN}°C)` });
    if (temp > TEMP_MAX) issues.push({ type: 'danger', text: `Suhu tinggi: ${temp.toFixed(1)}°C (max: ${TEMP_MAX}°C)` });
    if (hum < HUM_MIN) issues.push({ type: 'warning', text: `Kelembaban rendah: ${hum.toFixed(1)}% (min: ${HUM_MIN}%)` });
    if (hum > HUM_MAX) issues.push({ type: 'danger', text: `Kelembaban tinggi: ${hum.toFixed(1)}% (max: ${HUM_MAX}%)` });
    return issues;
  };

  const getWeatherColor = (rain: number, temp: number, hum: number) => {
    const isRainExtreme = rain < RAIN_MIN || rain > RAIN_MAX;
    const isTempExtreme = temp < TEMP_MIN || temp > TEMP_MAX;
    const isHumExtreme = hum < HUM_MIN || hum > HUM_MAX;

    const extremeCount = [isRainExtreme, isTempExtreme, isHumExtreme].filter(Boolean).length;

    if (extremeCount === 0) return "bg-emerald-100 border-emerald-200 text-emerald-800";
    if (extremeCount >= 2) return "bg-red-100 border-red-200 text-red-800";
    return "bg-amber-100 border-amber-200 text-amber-800";
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
      let hasData = false;
      let issues: any[] = [];

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
          hasData = true;
          rain = forecastDay.parameters?.RR?.forecast_value ?? 0;
          temp = forecastDay.parameters?.TAVG?.forecast_value ?? 0;
          hum = forecastDay.parameters?.RH_AVG?.forecast_value ?? 0;
          
          issues = getWeatherIssues(rain, temp, hum);
          bgColor = getWeatherColor(rain, temp, hum);
          icon = getWeatherIcon(rain, temp, hum);
        } else {
          bgColor = "bg-gray-50 border-gray-200 text-gray-800";
          icon = <Info className="w-3 h-3" />;
        }

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
        fullDate: format(date, "d MMMM yyyy"),
        type,
        bgColor,
        icon,
        rain,
        temp,
        hum,
        hasData,
        issues,
      };
    });

    // Statistik cuaca
    const plantingDays = gridDays.filter(d => d.type === "Masa Tanam" && d.hasData);
    const goodDays = plantingDays.filter(d => d.issues.length === 0).length;
    const warningDays = plantingDays.filter(d => d.issues.length === 1).length;
    const badDays = plantingDays.filter(d => d.issues.length >= 2).length;

    return (
      <Card className="mt-6">
        <CardHeader className="pb-3 sm:pb-4">
          <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
            <Calendar className="w-4 h-4 sm:w-5 sm:h-5" />
            Jadwal Penanaman
          </CardTitle>

          {/* Model indicator */}
          <div className="flex items-center gap-2 sm:gap-3 p-3 sm:p-4 bg-gradient-to-r from-teal-50 to-emerald-50 rounded-lg border border-teal-200">
            <Cloud className="w-4 h-4 sm:w-5 sm:h-5 text-teal-600 shrink-0" />
            <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-2">
              <span className="text-xs sm:text-sm font-medium text-teal-800">Model Prediksi:</span>
              <Badge 
                variant={currentModel === "lstm" ? "default" : "secondary"}
                className="w-fit bg-gradient-to-r from-teal-600 to-emerald-600 text-white text-xs"
              >
                {currentModel === "lstm" ? "LSTM Neural Network" : "Holt Winters"}
              </Badge>
              {isForecastLoading && (
                <span className="text-[10px] sm:text-xs text-teal-600 animate-pulse flex items-center gap-1">
                  <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-teal-600 rounded-full animate-bounce"></div>
                  Memuat data...
                </span>
              )}
            </div>
          </div>

          {/* Weather Statistics */}
          {plantingDays.length > 0 && (
            <Card className="bg-gradient-to-r from-teal-50 to-emerald-50 border-teal-200">
              <CardContent className="pt-3 sm:pt-4">
                <div className="flex items-center gap-2 mb-2 sm:mb-3">
                  <AlertTriangle className="w-3 h-3 sm:w-4 sm:h-4 text-teal-700" />
                  <span className="text-xs sm:text-sm font-semibold text-teal-900">Analisis Kondisi Cuaca</span>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="flex flex-col items-center p-3 bg-emerald-100 rounded-lg border border-emerald-200">
                    <CheckCircle className="w-5 h-5 text-emerald-600 mb-1" />
                    <span className="text-2xl font-bold text-emerald-800">{goodDays}</span>
                    <span className="text-xs text-emerald-600">Hari Ideal</span>
                  </div>
                  <div className="flex flex-col items-center p-3 bg-amber-100 rounded-lg border border-amber-200">
                    <AlertCircle className="w-5 h-5 text-amber-600 mb-1" />
                    <span className="text-2xl font-bold text-amber-800">{warningDays}</span>
                    <span className="text-xs text-amber-600">Hari Hati-hati</span>
                  </div>
                  <div className="flex flex-col items-center p-3 bg-red-100 rounded-lg border border-red-200">
                    <XCircle className="w-5 h-5 text-red-600 mb-1" />
                    <span className="text-2xl font-bold text-red-800">{badDays}</span>
                    <span className="text-xs text-red-600">Hari Berisiko</span>
                  </div>
                </div>
                <p className="text-xs text-gray-600 mt-3 text-center">
                  Total {plantingDays.length} hari masa tanam dengan data prediksi
                </p>
              </CardContent>
            </Card>
          )}

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
                  <TooltipContent side="top" className="max-w-sm">
                    <div className="space-y-2">
                      <p className="text-sm font-semibold border-b pb-1">
                        {item.fullDate} - {item.type}
                      </p>
                      {item.type === "Masa Tanam" && item.hasData && (
                        <div className="space-y-2">
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
                          </div>
                          
                          {item.issues.length > 0 && (
                            <div className="pt-2 border-t">
                              <div className="flex items-center gap-1 mb-1">
                                <AlertTriangle className="w-3 h-3 text-red-500" />
                                <span className="text-xs font-semibold text-red-600">Peringatan Cuaca:</span>
                              </div>
                              <ul className="space-y-1">
                                {item.issues.map((issue: any, i: number) => (
                                  <li key={i} className={`text-xs ${issue.type === 'danger' ? 'text-red-600' : 'text-amber-600'}`}>
                                    • {issue.text}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {item.issues.length === 0 && (
                            <div className="pt-2 border-t">
                              <div className="flex items-center gap-1 text-emerald-600">
                                <CheckCircle className="w-3 h-3" />
                                <span className="text-xs font-semibold">Kondisi ideal untuk penanaman</span>
                              </div>
                            </div>
                          )}
                          
                          <div className="flex items-center gap-2 text-xs text-gray-500 pt-1 border-t">
                            <Cloud className="w-3 h-3" />
                            <span>Model: {currentModel.toUpperCase()}</span>
                          </div>
                        </div>
                      )}
                      {item.type === "Masa Tanam" && !item.hasData && (
                        <div className="text-xs text-gray-500">
                          <Info className="w-3 h-3 inline mr-1" />
                          Data prediksi tidak tersedia untuk tanggal ini
                        </div>
                      )}
                      {(item.type === "Garap" || item.type === "Semai") && (
                        <div className="text-xs text-gray-600">
                          Periode persiapan lahan dan bibit sebelum penanaman
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
          <Card className="bg-gradient-to-r from-teal-50 to-emerald-50">
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
                <Cloud className="w-4 h-4 text-teal-600" />
                <span className="text-sm text-teal-700">
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
          <CardTitle className="flex items-center gap-2 text-xl">
            <Sprout className="w-5 h-5 text-green-600" />
            Perencanaan Penanaman Benih
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="space-y-2">
              <Label htmlFor="seed-select" className="flex items-center gap-2 text-base font-medium">
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
              <Label htmlFor="duration-input" className="flex items-center gap-2 text-base font-medium">
                <Clock className="w-4 h-4" />
                Durasi (Hari)
              </Label>
              <Input 
                id="duration-input"
                type="number" 
                placeholder="Durasi tanam" 
                value={duration} 
                onChange={(e) => setDuration(Number(e.target.value))}
                className="transition-all duration-200 focus:ring-2 focus:ring-teal-500"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="start-date" className="flex items-center gap-2 text-base font-medium">
                <Calendar className="w-4 h-4" />
                Tanggal Mulai Tanam
              </Label>
              <Input 
                id="start-date"
                type="date" 
                value={startDate} 
                onChange={(e) => setStartDate(e.target.value)}
                className="transition-all duration-200 focus:ring-2 focus:ring-teal-500"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="semai-status" className="flex items-center gap-2 text-base font-medium">
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
            className="mt-6 w-full sm:w-auto px-8 py-2 bg-gradient-to-r from-teal-600 to-emerald-600 hover:from-teal-700 hover:to-emerald-700 text-white text-base" 
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