"use client";

import { useState } from "react";
import { format } from "date-fns";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";

type SeedType = {
  name: string;
  growDuration: number; // dalam hari
};

const dummySeeds: SeedType[] = [
  { name: "Padi IR64", growDuration: 100 },
  { name: "Padi Inpari 32", growDuration: 105 },
  { name: "Padi Ciherang", growDuration: 95 },
];

export default function CalendarSeedPlanner() {
  const [selectedSeed, setSelectedSeed] = useState<SeedType | null>(null);
  const [startDate, setStartDate] = useState<string>("");

  const [showGrid, setShowGrid] = useState(false);

  const handleSubmit = () => {
    if (!selectedSeed || !startDate) return;
    setShowGrid(true);
  };

  const renderGrid = () => {
    if (!selectedSeed || !startDate) return null;

    const start = new Date(startDate);
    const totalDays = selectedSeed.growDuration;
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

  return (
    <div className="max-w-3xl mx-auto p-6 bg-white rounded-xl shadow">
      <h2 className="text-xl font-semibold mb-4">Perencanaan Tanam Harian</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <Label>Jenis Benih</Label>
          <Select
            onValueChange={(val) => {
              const seed = dummySeeds.find((s) => s.name === val);
              if (seed) setSelectedSeed(seed);
            }}
          >
            <SelectTrigger>
              <SelectValue placeholder="Pilih benih..." />
            </SelectTrigger>
            <SelectContent>
              {dummySeeds.map((seed) => (
                <SelectItem key={seed.name} value={seed.name}>
                  {seed.name}
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
