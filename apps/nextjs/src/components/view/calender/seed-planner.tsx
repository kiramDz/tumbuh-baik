"use client";

import { useState } from "react";
import { format } from "date-fns";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { getSeeds } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";

type SeedType = {
  _id: string;
  name: string;
  duration: number;
  createdAt: string;
  updatedAt: string;
};

type PreparationStatus = {
  isGarapDone: boolean;
  isSemaiDone: boolean;
};

export default function CalendarSeedPlanner() {
  const [selectedSeed, setSelectedSeed] = useState<SeedType | null>(null);
  const [startDate, setStartDate] = useState<string>("");
  const [showGrid, setShowGrid] = useState(false);
  const [preparationStatus, setPreparationStatus] = useState<PreparationStatus>({
    isGarapDone: false,
    isSemaiDone: false,
  });

  // Konstanta waktu persiapan (hari)
  const GARAP_DURATION = 5;
  const SEMAI_DURATION = 20;

  // Fetch seeds data from API
  const { data: seedsData } = useQuery({
    queryKey: ["get-seeds-planner"],
    queryFn: () => getSeeds(1, 100),
    refetchOnWindowFocus: false,
  });

  const handleSubmit = () => {
    if (!selectedSeed || !startDate) return;
    setShowGrid(true);
  };

  const calculateActualStartDate = () => {
    if (!startDate) return null;

    const targetPlantDate = new Date(startDate);
    let daysToSubtract = 0;

    // Hitung mundur berdasarkan status persiapan
    if (!preparationStatus.isSemaiDone) {
      daysToSubtract += SEMAI_DURATION;
    }
    if (!preparationStatus.isGarapDone) {
      daysToSubtract += GARAP_DURATION;
    }

    const actualStartDate = new Date(targetPlantDate);
    actualStartDate.setDate(actualStartDate.getDate() - daysToSubtract);

    return actualStartDate;
  };

  const renderGrid = () => {
    if (!selectedSeed || !startDate) return null;

    const actualStartDate = calculateActualStartDate();
    if (!actualStartDate) return null;

    const totalDays = selectedSeed.duration;

    
    let preparationDays = 0;
    if (!preparationStatus.isGarapDone) preparationDays += GARAP_DURATION;
    if (!preparationStatus.isSemaiDone) preparationDays += SEMAI_DURATION;

    
    const totalGridDays = preparationDays + totalDays;

    const gridDays = Array.from({ length: totalGridDays }, (_, i) => {
      const date = new Date(actualStartDate);
      date.setDate(date.getDate() + i);

      let type = "";
      let bgColor = "";

      if (i < preparationDays) {
        // Fase persiapan
        if (!preparationStatus.isGarapDone && !preparationStatus.isSemaiDone) {
          // Belum garap dan belum semai
          if (i < GARAP_DURATION) {
            type = "garap";
            bgColor = "bg-orange-200";
          } else {
            type = "semai";
            bgColor = "bg-blue-200";
          }
        } else if (!preparationStatus.isGarapDone && preparationStatus.isSemaiDone) {
          // Belum garap tapi sudah semai
          type = "garap";
          bgColor = "bg-orange-200";
        } else if (preparationStatus.isGarapDone && !preparationStatus.isSemaiDone) {
          // Sudah garap tapi belum semai
          type = "semai";
          bgColor = "bg-blue-200";
        }
      } else {
        // Fase tanam hingga panen
        const dayInCycle = i - preparationDays;
        if (dayInCycle < totalDays - 20) {
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
            Estimasi panen: <strong>{format(new Date(new Date(startDate).getTime() + (selectedSeed.duration - 20) * 24 * 60 * 60 * 1000), "d MMM yyyy")}</strong>
          </p>
        </div>
      </div>
    );
  };

  const seeds = seedsData?.items || [];

  return (
    <div className="mx-auto bg-white">
      <h2 className="text-xl font-semibold mb-4">Perencanaan Tanam Harian</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <Label>Jenis Benih</Label>
          <Select
            onValueChange={(val) => {
              const seed = seeds.find((s: SeedType) => s._id === val);
              if (seed) setSelectedSeed(seed);
            }}
          >
            <SelectTrigger>
              <SelectValue placeholder="Pilih benih..." />
            </SelectTrigger>
            <SelectContent>
              {seeds.map((seed: SeedType) => (
                <SelectItem key={seed._id} value={seed._id}>
                  {seed.name} ({seed.duration} hari)
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label>Status Garap Sawah</Label>
          <Select
            onValueChange={(val) =>
              setPreparationStatus((prev) => ({
                ...prev,
                isGarapDone: val === "sudah",
              }))
            }
          >
            <SelectTrigger>
              <SelectValue placeholder="Pilih status..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="belum">Belum Garap</SelectItem>
              <SelectItem value="sudah">Sudah Garap</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label>Status Semai</Label>
          <Select
            onValueChange={(val) =>
              setPreparationStatus((prev) => ({
                ...prev,
                isSemaiDone: val === "sudah",
              }))
            }
          >
            <SelectTrigger>
              <SelectValue placeholder="Pilih status..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="belum">Belum Semai</SelectItem>
              <SelectItem value="sudah">Sudah Semai</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label>Target Tanggal Mulai Tanam</Label>
          <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        </div>
      </div>

      <Button onClick={handleSubmit} className="mt-4">
        Buat Perkiraan
      </Button>

      {showGrid && renderGrid()}
    </div>
  );
}
