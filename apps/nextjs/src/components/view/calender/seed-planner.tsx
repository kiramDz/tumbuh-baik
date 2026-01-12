"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getSeeds, createSeed, getHoltWinterDaily } from "@/lib/fetch/files.fetch";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Combobox } from "@/components/combobox";
import { format, addDays } from "date-fns";
import { Label } from "@/components/ui/label";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";

interface SeedItem {
  name: string;
  duration: number;
}

const RAIN_MIN = 5.7;
const RAIN_MAX = 16.7;
const TEMP_MIN = 24;
const TEMP_MAX = 29;
const HUM_MIN = 33;
const HUM_MAX = 90;

// === BARU === Durasi untuk fase pencabutan dan penaburan bibit (tetap)
const TABUR_DURATION = 2;

export default function CalendarSeedPlanner() {
  const queryClient = useQueryClient();
  const [selectedSeedName, setSelectedSeedName] = useState<string>("");
  const [duration, setDuration] = useState<number>(10);
  const [startDate, setStartDate] = useState<string>("");

  // === BARU === State untuk durasi garap sawah, default 10 hari
  const [garapDurationInput, setGarapDurationInput] = useState<number>(10);
  const [semaiDurationInput, setSemaiDurationInput] = useState<number>(20);
  const [showGrid, setShowGrid] = useState(false);

  const { data: seedsData } = useQuery({
    queryKey: ["get-seeds-planner"],
    queryFn: () => getSeeds(1, 100),
    refetchOnWindowFocus: false,
  });

  const { data: forecastData } = useQuery({
    queryKey: ["holt-winter-all"],
    queryFn: () => getHoltWinterDaily(1, 731),
  });

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

  // === DIHAPUS === Fungsi calculateActualStartDate tidak lagi diperlukan

  const getWeatherColor = (rain: number, temp: number, hum: number) => {
    const isRainSesuai = rain >= RAIN_MIN && rain <= RAIN_MAX;
    const isTempSesuai = temp >= TEMP_MIN && temp <= TEMP_MAX;
    const isHumSesuai = hum >= HUM_MIN && hum <= HUM_MAX;

    const sesuaiCount = Number(isRainSesuai) + Number(isTempSesuai) + Number(isHumSesuai);

    if (sesuaiCount === 3) return "bg-green-300";
    if (sesuaiCount === 2) return "bg-green-100";
    return "bg-red-300";
  };

  const renderGrid = () => {
    if (!selectedSeedName || !duration || !startDate) return null;

    // === DIUBAH === Logika Kalkulasi Jadwal Maju
    const projectStartDate = new Date(startDate);

    // Durasi persiapan adalah waktu terpanjang antara garap dan semai
    const preparationDuration = Math.max(garapDurationInput, semaiDurationInput);

    // Tanggal tanam di sawah adalah setelah fase persiapan dan fase tabur selesai
    const actualPlantDate = addDays(projectStartDate, preparationDuration + TABUR_DURATION);

    // Total hari dari awal persiapan hingga panen selesai
    const totalGridDays = preparationDuration + TABUR_DURATION + duration;

    const gridDays = Array.from({ length: totalGridDays }, (_, i) => {
      const date = addDays(projectStartDate, i);
      let type = "";
      let bgColor = "";
      let rain = 0,
        temp = 0,
        hum = 0;

      const tanamStartIndex = preparationDuration + TABUR_DURATION;

      if (i < preparationDuration) {
        // Fase Persiapan (Garap & Semai berjalan paralel)
        const prepActivities = [];
        if (i < garapDurationInput) prepActivities.push("Garap");
        if (i < semaiDurationInput) prepActivities.push("Semai");
        type = prepActivities.join(" & ");
        bgColor = "bg-orange-200";
      } else if (i < tanamStartIndex) {
        // Fase Tabur Bibit
        type = "Cabut & Tabur Bibit";
        bgColor = "bg-cyan-200"; // Warna baru untuk fase tabur
      } else {
        // Fase Tanam di Sawah
        type = "Masa Tanam";
        const forecastDay = forecastData.items.find((f: any) => format(new Date(f.forecast_date), "yyyy-MM-dd") === format(date, "yyyy-MM-dd"));

        if (forecastDay) {
          rain = forecastDay.parameters?.RR_imputed?.forecast_value ?? 0;
          temp = forecastDay.parameters?.TAVG?.forecast_value ?? 0;
          hum = forecastDay.parameters?.RH_AVG_preprocessed?.forecast_value ?? 0;
          bgColor = getWeatherColor(rain, temp, hum);
        } else {
          bgColor = "bg-gray-200";
        }

        const dayInGrowingCycle = i - tanamStartIndex;
        if (dayInGrowingCycle >= duration - 20) {
          type = "Panen";
          bgColor = "bg-yellow-300";
        }
      }

      return { day: i + 1, date: format(date, "MMM d"), type, bgColor, rain, temp, hum };
    });

    return (
      <div className="mt-6">
        {/* === DIUBAH === Legenda disesuaikan dengan fase baru */}
        <div className="mb-4 flex flex-wrap gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-orange-200 rounded"></div>
            <span>Persiapan (Garap & Semai)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-cyan-200 rounded"></div>
            <span>Cabut & Tabur Bibit ({TABUR_DURATION} hari)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-300 rounded"></div>
            <span>Sangat Cocok Tanam</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-100 rounded"></div>
            <span>Cukup Cocok Tanam</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-300 rounded"></div>
            <span>Tidak Cocok Tanam</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-300 rounded"></div>
            <span>Panen</span>
          </div>
        </div>
        <TooltipProvider>
          <div className="grid grid-cols-6 gap-2 text-sm">
            {gridDays.map((item, idx) => (
              <Tooltip key={idx}>
                <TooltipTrigger asChild>
                  <div className={`p-2 rounded-lg text-center ${item.bgColor}`}>
                    <div className="font-semibold">{item.date}</div>
                    <div className="text-xs">{item.type}</div>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p className="text-xs">
                    Curah hujan: {item.rain.toFixed(2)} mm <br />
                    Suhu: {item.temp.toFixed(2)} °C <br />
                    Kelembaban: {item.hum.toFixed(2)} %
                  </p>
                </TooltipContent>
              </Tooltip>
            ))}
          </div>
        </TooltipProvider>

        {/* === DIUBAH === Ringkasan jadwal disesuaikan dengan logika baru */}
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold mb-2">Ringkasan Jadwal:</h3>
          <p className="text-sm text-gray-600">
            Mulai persiapan (Garap/Semai): <strong>{format(projectStartDate, "d MMM yyyy")}</strong>
          </p>
          <p className="text-sm text-gray-600">
            Mulai tanam di sawah: <strong>{format(actualPlantDate, "d MMM yyyy")}</strong>
          </p>
          <p className="text-sm text-gray-600">
            Estimasi panen mulai: <strong>{format(addDays(actualPlantDate, duration - 20), "d MMM yyyy")}</strong>
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* === DIUBAH === Penambahan input durasi garap & perubahan label */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="flex flex-col gap-2">
          <Label htmlFor="benih-choice">Benih</Label>
          <Combobox  options={seedsData?.items.map((s: SeedItem) => s.name) || []} value={selectedSeedName} onValueChange={handleSeedChange} />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="benih-duration">Durasi Benih (hari)</Label>
          <Input id="benih-duration" type="number" placeholder="10" value={duration} onChange={(e) => setDuration(Number(e.target.value))} />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="garap-duration">Durasi Garap Sawah (hari)</Label>
          <Input id="garap-duration" type="number" placeholder="Contoh: 10" value={garapDurationInput} onChange={(e) => setGarapDurationInput(Number(e.target.value))} min="0" />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="semai-duration">Durasi Semai (hari)</Label>
          <Input id="semai-duration" type="number" placeholder="Contoh: 20" value={semaiDurationInput} onChange={(e) => setSemaiDurationInput(Number(e.target.value))} min="0" />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="start-date">Tgl Mulai Persiapan Lahan</Label>
          <Input id="start-date" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        </div>
      </div>

      <Button className="mt-2" onClick={handleSubmit} disabled={!selectedSeedName || !duration || !startDate}>
        Buat Jadwal
      </Button>

      {showGrid && renderGrid()}
    </div>
  );
}
