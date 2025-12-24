"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import axios from "axios";
import { Calendar1 } from "lucide-react";
import { Dialog, DialogContent, DialogTrigger, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "sonner";
import { createForecastConfig } from "@/lib/fetch/files.fetch";
import { format } from "date-fns";
import { id } from "date-fns/locale";
import { cn } from "@/lib/utils";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

interface ForecastDialogProps {
  onSubmit?: () => void;
}

interface DatasetMeta {
  collectionName: string;
  name?: string;
  columns?: string[];
}

export function ForecastDialog({ onSubmit }: ForecastDialogProps) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [selected, setSelected] = useState<{ collectionName: string; columnName: string }[]>([]);
  const [startDate, setStartDate] = useState<Date>();
  const queryClient = useQueryClient();

  const { data: datasetList = [] } = useQuery<DatasetMeta[]>({
    queryKey: ["dataset-meta"],
    queryFn: async () => {
      const res = await axios.get("/api/v1/dataset-meta");
      return res.data?.data || [];
    },
  });

  const { mutate, isPending } = useMutation({
    mutationKey: ["forecast-config"],
    mutationFn: createForecastConfig,
    onSuccess: () => {
      toast.success("Konfigurasi berhasil disimpan");
      queryClient.invalidateQueries({ queryKey: ["forecast-config"] });
      onSubmit?.();
      setOpen(false);
      setName("");
      setSelected([]);
      setStartDate(undefined);
    },
    onError: (error) => {
      console.error("Error saving forecast config:", error);
      toast.error("Gagal menyimpan konfigurasi");
    },
  });
  const handleCheckbox = (collectionName: string, columnName: string, checked: boolean) => {
    if (checked) {
      setSelected((prev) => [...prev, { collectionName, columnName }]);
    } else {
      setSelected((prev) => prev.filter((item) => !(item.collectionName === collectionName && item.columnName === columnName)));
    }
  };

  const handleSave = () => {
    if (!name.trim() || selected.length === 0) {
      return toast.error("Nama konfigurasi dan kolom wajib diisi");
    }

    if (!startDate) {
      return toast.error("Tanggal mulai peramalan wajib diisi");
    }

    mutate({
      name: name.trim(),
      columns: selected,
      startDate: format(startDate, "yyyy-MM-dd"), // Format: "2025-01-15"
    });
  };

  const endDate = startDate ? new Date(startDate.getFullYear() + 1, startDate.getMonth(), startDate.getDate()) : null;

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="default">Tambah Kolom</Button>
      </DialogTrigger>

      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Tambah Konfigurasi Peramalan</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <Label>Nama Konfigurasi</Label>
            <Input placeholder="Contoh: Peramalan Suhu & NDVI" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div>
            <Label>Tanggal Mulai Peramalan</Label>
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className={cn("w-full justify-start text-left font-normal", !startDate && "text-muted-foreground")}>
                  <Calendar1 className="mr-2 h-4 w-4" />
                  {startDate ? format(startDate, "PPP", { locale: id }) : "Pilih tanggal mulai"}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar mode="single" selected={startDate} onSelect={setStartDate} initialFocus locale={id} />
              </PopoverContent>
            </Popover>

            {/* Preview Tanggal Akhir */}
            {endDate && (
              <p className="text-sm text-muted-foreground mt-2">
                Peramalan akan dilakukan hingga: <span className="font-semibold">{format(endDate, "PPP", { locale: id })}</span>
              </p>
            )}
          </div>

          <div className="space-y-4 max-h-96 overflow-y-auto">
            {datasetList.map((dataset) => (
              <div key={dataset.collectionName}>
                <Label className="font-semibold">{dataset.name || dataset.collectionName}</Label>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  {dataset.columns?.map((col: string) => {
                    const isChecked = selected.some((item) => item.collectionName === dataset.collectionName && item.columnName === col);

                    return (
                      <label key={col} className="flex items-center space-x-2 cursor-pointer">
                        <Checkbox checked={isChecked} onCheckedChange={(checked) => handleCheckbox(dataset.collectionName, col, !!checked)} />
                        <span className="text-sm">{col}</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          {selected.length > 0 && (
            <div className="mt-4 p-3 bg-gray-50 rounded-md">
              <Label className="font-semibold">Kolom yang dipilih:</Label>
              <div className="mt-2 space-y-1">
                {selected.map((item, index) => (
                  <div key={index} className="text-sm text-gray-600">
                    {item.collectionName} → {item.columnName}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex justify-end gap-2 pt-4">
            <Button variant="outline" onClick={() => setOpen(false)}>
              Batal
            </Button>
            <Button onClick={handleSave} disabled={isPending}>
              {isPending ? "Menyimpan..." : "Simpan"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
