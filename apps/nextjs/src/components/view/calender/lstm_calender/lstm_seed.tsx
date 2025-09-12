import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getSeeds, createSeed, getLSTMDaily } from "@/lib/fetch/files.fetch";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Combobox } from "@/components/combobox";
import { format } from "date-fns";
import { Label } from "@/components/ui/label";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";


interface SeedItem {
    name: string;
    duration: number;
}

const MIN_RAIN = 5.7;
const MAX_RAIN = 16.7;
const MIN_TEMP = 24;
const MAX_TEMP = 29;
const MIN_HUM = 33;
const MAX_HUM = 90;

const GARAP_DURATION = 5;
const SEMAI_DURATION = 20;

export default function LSTMCalendarSeedPlanner() {
    const queryClient = useQueryClient();
    const [selectedSeedName, setSelectedSeedName] = useState<string>("");
    const [duration, setDuration] = useState<number>(0);
    const [startDate, setStartDate] = useState<string>("");
    const [semaiStatus, setSemaiStatus] = useState<string>("belum");
    const [showGrid, setShowGrid] = useState(false);

    const { data: seedsData } = useQuery({
        queryKey: ["get-seeds-planner"],
        queryFn: () => getSeeds(1, 100),
        refetchOnWindowFocus: false,
    });

    const { data: forecastData } = useQuery({
        queryKey: ["lstm-daily-all"],
        queryFn: () => getLSTMDaily(1, 731), // Ambil semua data 2 tahun
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

    const calculateActualStartDate = () => {
        if (!startDate) return null;
        const targetPlantDate = new Date(startDate);
        const daysToSubtract = semaiStatus === "belum" ? GARAP_DURATION + SEMAI_DURATION : GARAP_DURATION;
        const actualStartDate = new Date(targetPlantDate);
        actualStartDate.setDate(targetPlantDate.getDate() - daysToSubtract);
        return actualStartDate;
    };

    const getWeatherColor = (rain: number, temp: number, hum: number) => {
        const rainExtreme = rain < MIN_RAIN || rain > MAX_RAIN;
        const tempExtreme = temp < MIN_TEMP || temp > MAX_TEMP;
        const humExtreme = hum < MIN_HUM || hum > MAX_HUM;

        const extremeCount = [rainExtreme, tempExtreme, humExtreme].filter(Boolean).length;

        if (extremeCount === 0) return "bg-green-300";
        if (extremeCount === 1) return "bg-red-300";
        return "bg-green-100";
    };

    const renderGrid = () => {
        if (!selectedSeedName || !duration || !startDate) return null;
        const actualStartDate = calculateActualStartDate();
        if (!actualStartDate) return null;

        const preparationDays = semaiStatus === "belum" ? GARAP_DURATION + SEMAI_DURATION : GARAP_DURATION;
        const totalDays = preparationDays + duration;

        const dayGrid = Array.from({ length: totalDays }, (_, i) => {
            const date = new Date(actualStartDate);
            date.setDate(actualStartDate.getDate() + i);

            let type = "";
            let bgColor = "";
            let rain = 0, temp = 0, hum = 0;

            if (i < preparationDays) {
                if (semaiStatus === "belum") {
                    type = i < GARAP_DURATION ? "Garap" : "Semai";
                    bgColor = i < GARAP_DURATION ? "bg-yellow-200" : "bg-blue-200";
                } else {
                    type = "Garap";
                    bgColor = "bg-yellow-200";
                }
            } else {
                type = "Masa Tanam";
                const forecastDay = forecastData?.items.find((f: any) => format(new Date(f.date), "yyyy-MM-dd") === format(date, "yyyy-MM-dd"));

                if (forecastDay) {
                    rain = forecastDay.parameters?.RR_imputed?.forecast_value ?? 0;
                    temp = forecastDay.paramaters?.TAVG?.forecast_value ?? 0;
                    hum = forecastDay.parameters?.RH_AVG.preporecessed?.forecast_value ?? 0;
                    bgColor = getWeatherColor(rain, temp, hum);
                } else {
                    bgColor = "bg-gray-200";
                }
                
                const dayInCycle = i - preparationDays;
                if (dayInCycle >= duration - 20) {
                    type = "Masa Panen";
                    bgColor = "bg-yellow-300";
                }
            }

            return {
                day: i + 1,
                date: format(date, "MMM d"),
                type,
                bgColor,
                rain,
                temp,
                hum,
            };
        });

        return (
            <div className="mt-6">
                <div className="mb-4 flex flex-wrap gap-4 text-sm">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-orange-200 rounded"></div>
                        <span>Garap Sawah ({GARAP_DURATION} hari)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-blue-200 rounded"></div>
                        <span>Semai ({SEMAI_DURATION} hari)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-green-300 rounded"></div>
                        <span>Masa Tanam</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-yellow-300 rounded"></div>
                        <span>Panen</span>
                    </div>
                </div>
                <TooltipProvider>
                    <div className="grid grid-cols-6 gap-2 text-sm">
                        {dayGrid.map((item, idx) => (
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

                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                    <h3 className="font-semibold mb-2">Ringkasan Jadwal:</h3>
                    <p className="text-sm text-gray-60">
                        Mulai Persiapan: <strong>{format(actualStartDate, "MMM d")}</strong>
                    </p>
                    <p className="text-sm text-gray-600">
                        Target Mulai Tanam: <strong>{format(new Date(startDate), "MMM d")}</strong>
                    </p>
                    <p className="text-sm text-gray-600">
                        Estimasi Panen: <strong>{format(new Date(new Date(startDate).getTime() + (duration - 20) * 24 * 60 * 60 * 1000), "d MMM yyyy")}</strong>
                    </p>
                </div>
            </div>
        );
    };

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex flex-col gap-2">
                    <Label htmlFor="picture">Benih</Label>
                    <Combobox options={seedsData?.items.map((s: SeedItem) => s.name) || []} value={selectedSeedName} onValueChange={handleSeedChange} />
                </div>
                <div className="flex flex-col gap-2">
                    <Label htmlFor="benih">Durasi Benih</Label>
                    <Input type="number" placeholder="Durasi benih dalam hari" value={duration} onChange={(e) => setDuration(Number(e.target.value))} />
                </div>
                <div className="flex flex-col gap-2">
                    <Label htmlFor="tanggal">Tanggal Mulai</Label>
                    <Input type="date" placeholder="Tanggal Mulai Tanam" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
                </div>
                <div className="flex flex-col gap-2">
                    <Label htmlFor="semai">Status Semai</Label>
                    <Select value={semaiStatus} onValueChange={setSemaiStatus}>
                        <SelectTrigger>
                            <SelectValue placeholder="Status Semai" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="belum">Belum Semai</SelectItem>
                            <SelectItem value="sudah">Sudah Semai</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
            </div>

            <Button className="mt-2" onClick={handleSubmit} disabled={!selectedSeedName || !duration || !startDate}>
                Buat Jadwal
            </Button>

            {showGrid && renderGrid()}
        </div>
    );
}