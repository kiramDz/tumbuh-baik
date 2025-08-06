"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { AddDatasetMeta } from "@/lib/fetch/files.fetch";
import { parseFile } from "@/lib/parse-upload";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";

//TODO : Buat useffect setip kli ada data baru yg ditambah, biar auto reload
export default function AddDatasetDialog() {
  const queryClient = useQueryClient();

  const [form, setForm] = useState({
    name: "",
    source: "",
    collectionName: "",
    description: "",
    status: "raw",
  });

  const [file, setFile] = useState<File | null>(null);
  const [open, setOpen] = useState(false);

  const { mutate, isPending } = useMutation({
    mutationKey: ["add-dataset"],
    mutationFn: AddDatasetMeta,
    onSuccess: () => {
      toast.success("Dataset berhasil ditambahkan");
      queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
      setOpen(false);
      setForm({
        name: "",
        source: "",
        collectionName: "",
        description: "",
        status: "raw",
      });
      setFile(null);
    },
    onError: () => {
      toast.error("Gagal menambahkan dataset");
    },
  });

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!form.name || !form.source || !file) {
      return toast.error("Mohon lengkapi semua data wajib");
    }

    const fileType = file.name.endsWith(".json")
      ? "json"
      : file.name.endsWith(".csv")
      ? "csv"
      : null;
    if (!fileType) return toast.error("Hanya file CSV atau JSON yang didukung");

    const buffer = await file.arrayBuffer();
    const parsed = await parseFile({
      fileBuffer: Buffer.from(buffer),
      fileType,
    });

    mutate({
      name: form.name,
      source: form.source,
      fileType,
      collectionName: form.collectionName,
      description: form.description,
      status: form.status,
      records: parsed,
    });
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline">Tambah Dataset</Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Tambah Dataset</DialogTitle>
          <DialogDescription>
            Unggah file CSV/JSON beserta metadata singkat.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="name">Nama Dataset</Label>
            <Input
              id="name"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              required
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="source">Sumber</Label>
            <select
              id="source"
              value={form.source}
              onChange={(e) => setForm({ ...form, source: e.target.value })}
              className="border rounded px-3 py-2"
              required
            >
              <option value="">Pilih sumber data...</option>
              <option value="Data BMKG (https://dataonline.bmkg.go.id/)">
                Data BMKG (https://dataonline.bmkg.go.id/)
              </option>
              <option value="Data NASA (https://power.larc.nasa.gov/)">
                Data NASA (https://power.larc.nasa.gov/)
              </option>
              <option value="Data Satelit">
                NDVI LSWI EVI Google Earth Engine
              </option>
            </select>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="collectionName">Nama Koleksi (Opsional)</Label>
            <Input
              id="collectionName"
              value={form.collectionName}
              onChange={(e) =>
                setForm({ ...form, collectionName: e.target.value })
              }
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="status">Status</Label>
            <select
              id="status"
              value={form.status}
              onChange={(e) => setForm({ ...form, status: e.target.value })}
              className="border rounded px-3 py-2"
            >
              <option value="raw">Raw</option>
              <option value="cleaned">Cleaned</option>
              <option value="validated">Validated</option>
            </select>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="description">Deskripsi</Label>
            <Input
              id="description"
              value={form.description}
              onChange={(e) =>
                setForm({ ...form, description: e.target.value })
              }
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="file">Upload File</Label>
            <Input
              id="file"
              type="file"
              accept=".csv,.json"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              required
            />
          </div>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline">Batal</Button>
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
