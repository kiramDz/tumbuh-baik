"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Dialog, DialogContent, DialogTrigger, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogClose } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { createBmkgData } from "@/lib/fetch/files.fetch";

export default function DataTableAddDialog() {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [formState, setFormState] = useState<Record<string, string>>({});

  const { mutate, isPending } = useMutation({
    mutationKey: ["add-bmkg"],
    mutationFn: createBmkgData,
    onSuccess: () => {
      toast.success("Data BMKG berhasil ditambahkan");
      queryClient.invalidateQueries({ queryKey: ["bmkg"] });
      setOpen(false);
      setFormState({});
    },
    onError: () => {
      toast.error("Gagal menambahkan data BMKG");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const payload = {
        ...formState,
        Date: new Date(formState.Date),
        Year: Number(formState.Year),
        Month: String(formState.Month),
        Day: Number(formState.Day),
        TN: Number(formState.TN),
        TX: Number(formState.TX),
        TAVG: Number(formState.TAVG),
        RH_AVG: Number(formState.RH_AVG),
        RR: Number(formState.RR),
        SS: Number(formState.SS),
        FF_X: Number(formState.FF_X),
        DDD_X: Number(formState.DDD_X),
        FF_AVG: Number(formState.FF_AVG),
        DDD_CAR: formState.DDD_CAR,
        Season: formState.Season,
        is_RR_missing: Number(formState.is_RR_missing),
      };
      mutate(payload);
    } catch {
      toast.error("Format data tidak valid");
    }
  };

  const handleChange = (key: string, value: string) => {
    setFormState((prev) => ({ ...prev, [key]: value }));
  };

  const editableKeys = ["Date", "Year", "Month", "Day", "TN", "TX", "TAVG", "RH_AVG", "RR", "SS", "FF_X", "DDD_X", "FF_AVG", "DDD_CAR", "Season", "is_RR_missing"];

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline">Tambah Data</Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[600px] max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Tambah Data BMKG</DialogTitle>
          <DialogDescription>Masukkan data secara manual untuk 1 hari.</DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          {editableKeys.map((key) => (
            <div key={key} className="grid gap-2">
              <Label htmlFor={key}>{key}</Label>
              <Input id={key} type={key === "Date" ? "date" : "text"} required value={formState[key] || ""} onChange={(e) => handleChange(key, e.target.value)} />
            </div>
          ))}
          <DialogFooter className="mt-4">
            <DialogClose asChild>
              <Button variant="outline" type="button">
                Batal
              </Button>
            </DialogClose>
            <Button type="submit" disabled={isPending}>
              {isPending ? "Menyimpan..." : "Simpan"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
