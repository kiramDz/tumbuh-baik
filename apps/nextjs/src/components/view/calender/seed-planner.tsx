import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getSeeds, createSeed } from "@/lib/fetch/files.fetch";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Combobox } from "@/components/combobox";
import { format } from "date-fns";
import { Label } from "@/components/ui/label";

export default function CalendarSeedPlanner() {
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

  const createSeedMutation = useMutation({
    mutationFn: createSeed,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["get-seeds-planner"] }),
  });

  const GARAP_DURATION = 5;
  const SEMAI_DURATION = 20;

  const handleSeedChange = (name: string) => {
    setSelectedSeedName(name);
    const match = seedsData?.items.find((s) => s.name.toLowerCase() === name.toLowerCase());
    if (match) setDuration(match.duration);
  };

  const handleSubmit = async () => {
    if (!selectedSeedName || !duration || !startDate) return;

    const exists = seedsData?.items.find((s) => s.name.toLowerCase() === selectedSeedName.toLowerCase() && s.duration === duration);
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

  const renderGrid = () => {
    if (!selectedSeedName || !duration || !startDate) return null;
    const actualStartDate = calculateActualStartDate();
    if (!actualStartDate) return null;

    const preparationDays = semaiStatus === "belum" ? GARAP_DURATION + SEMAI_DURATION : GARAP_DURATION;
    const totalGridDays = preparationDays + duration;

    const gridDays = Array.from({ length: totalGridDays }, (_, i) => {
      const date = new Date(actualStartDate);
      date.setDate(date.getDate() + i);

      let type = "";
      let bgColor = "";

      if (i < preparationDays) {
        if (semaiStatus === "belum") {
          type = i < GARAP_DURATION ? "garap" : "semai";
          bgColor = i < GARAP_DURATION ? "bg-orange-200" : "bg-blue-200";
        } else {
          type = "garap";
          bgColor = "bg-orange-200";
        }
      } else {
        const dayInCycle = i - preparationDays;
        if (dayInCycle < duration - 20) {
          type = "masa tanam";
          bgColor = "bg-green-300";
        } else {
          type = "panen";
          bgColor = "bg-yellow-300";
        }
      }

      return {
        day: i + 1,
        date: format(date, "MMM d"),
        type,
        bgColor,
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

        <div className="grid grid-cols-6 gap-2 text-sm">
          {gridDays.map((item, idx) => (
            <div key={idx} className={`p-2 rounded-lg text-center ${item.bgColor}`}>
              <div className="font-semibold">{item.date}</div>
              <div className="text-xs">{item.type}</div>
            </div>
          ))}
        </div>

        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold mb-2">Ringkasan Jadwal:</h3>
          <p className="text-sm text-gray-600">
            Mulai persiapan: <strong>{format(actualStartDate, "d MMM yyyy")}</strong>
          </p>
          <p className="text-sm text-gray-600">
            Target mulai tanam: <strong>{format(new Date(startDate), "d MMM yyyy")}</strong>
          </p>
          <p className="text-sm text-gray-600">
            Estimasi panen: <strong>{format(new Date(new Date(startDate).getTime() + (duration - 20) * 24 * 60 * 60 * 1000), "d MMM yyyy")}</strong>
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
          <Combobox options={seedsData?.items.map((s) => s.name) || []} value={selectedSeedName} onValueChange={handleSeedChange} />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="benih">Durasi Benih</Label>
          <Input type="number" placeholder="Durasi (hari)" value={duration} onChange={(e) => setDuration(Number(e.target.value))} />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="tanggal">Tanggal Mulai</Label>
          <Input type="date" placeholder="Tanggal Mulai Tanam" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="semai">Status semai</Label>
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
