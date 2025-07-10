"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import axios from "axios";
import { Dialog, DialogContent, DialogTrigger, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "sonner";
import { createForecastConfig } from "@/lib/fetch/files.fetch";

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

    mutate({ name: name.trim(), columns: selected });
  };

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
