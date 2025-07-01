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
  duration: number; // dalam hari
  createdAt: string;
  updatedAt: string;
};

export default function CalendarSeedPlanner() {
  const [selectedSeed, setSelectedSeed] = useState<SeedType | null>(null);
  const [startDate, setStartDate] = useState<string>("");
  const [showGrid, setShowGrid] = useState(false);

  // Fetch seeds data from API
  const {
    data: seedsData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["get-seeds-planner"],
    queryFn: () => getSeeds(1, 100), // ambil semua data untuk dropdown
    refetchOnWindowFocus: false,
  });

  const handleSubmit = () => {
    if (!selectedSeed || !startDate) return;
    setShowGrid(true);
  };

  const renderGrid = () => {
    if (!selectedSeed || !startDate) return null;

    const start = new Date(startDate);
    const totalDays = selectedSeed.duration;
    const gridDays = Array.from({ length: totalDays }, (_, i) => {
      const date = new Date(start);
      date.setDate(date.getDate() + i);
      return {
        day: i + 1,
        date: format(date, "MMM d"),
        type: i < 10 ? "persiapan" : i < totalDays - 20 ? "masa tanam" : "panen",
      };
    });

    return (
      <div className="mt-6 grid grid-cols-6 gap-2 text-sm">
        {gridDays.map((item, idx) => (
          <div key={idx} className={`p-2 rounded-lg text-center ${item.type === "persiapan" ? "bg-gray-200" : item.type === "masa tanam" ? "bg-green-300" : "bg-yellow-300"}`}>
            <div className="font-semibold">{item.date}</div>
            <div>{item.type}</div>
          </div>
        ))}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="max-w-3xl mx-auto p-6 bg-white rounded-xl shadow">
        <div className="text-center">Loading seeds data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-3xl mx-auto p-6 bg-white rounded-xl shadow">
        <div className="text-center text-red-500">Error loading seeds data</div>
      </div>
    );
  }

  const seeds = seedsData?.items || [];

  return (
    <div className="max-w-3xl mx-auto p-6 bg-white rounded-xl shadow">
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
          <Label>Tanggal Mulai Tanam</Label>
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
