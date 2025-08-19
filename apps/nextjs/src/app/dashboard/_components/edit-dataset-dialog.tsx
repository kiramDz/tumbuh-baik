"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import React, { useState } from "react";
import { UpdateDatasetMeta, DeleteDatasetMeta } from "@/lib/fetch/files.fetch";
import { Button } from "@/components/ui/button";
import { Menu } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogTrigger,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Trash } from "lucide-react";
import { toast } from "sonner";

interface EditDatasetDialogProps {
  dataset: {
    _id: string;
    name: string;
    source: string;
    collectionName?: string;
    description?: string;
    status: string;
  };
}

export default function EditDatasetDialog({ dataset }: EditDatasetDialogProps) {
  console.log("🔍 Dataset received in EditDatasetDialog:", dataset);
  console.log("🔍 Dataset ID:", dataset?._id);
  console.log("🔍 Dataset name:", dataset?.name);
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({
    name: dataset.name,
    source: dataset.source,
    // name: dataset.name || "",
    // source: dataset.source || "",
    collectionName: dataset.collectionName || "",
    description: dataset.description || "",
    status: dataset.status || "raw",
  });
  // Mutation update
  const { mutate: updateDataset, isPending: isUpdating } = useMutation({
    mutationKey: ["update-dataset", dataset._id],
    mutationFn: (data: typeof form) => UpdateDatasetMeta(dataset._id, data),
    onSuccess: () => {
      toast.success("Dataset berhasil diperbarui");
      queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
      setOpen(false);
    },
    onError: () => {
      toast.error("Gagal memperbarui dataset");
    },
  });
  // Mutation delete
  const { mutate: deleteDataset, isPending: isDeleting } = useMutation({
    mutationKey: ["delete-dataset", dataset._id],
    mutationFn: () => DeleteDatasetMeta(dataset._id),
    onSuccess: () => {
      toast.success("Dataset berhasil dihapus");
      queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
      setOpen(false);
    },
    onError: () => {
      toast.error("Gagal menghapus dataset");
    },
  });
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    updateDataset(form);
  };
  const handleDelete = () => {
    if (confirm("Yakin ingin menghapus dataset ini?")) {
      deleteDataset();
    }
  };
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <button className="group/menu flex h-8 w-8 items-center justify-center rounded-full bg-gray-100 transition-all duration-200 hover:bg-black hover:scale-110">
          <Menu className="h-5 w-5 text-gray-600 transition-colors duration-200 group-hover/menu:text-white" />
        </button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Edit Dataset</DialogTitle>
          <DialogDescription>
            Ubah metadata dataset atau hapus dataset.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="name">Nama Dataset</Label>
            {/* <Input
              id="name"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              required
            /> */}
            <Input
              id="name"
              value={form.name}
              onChange={(e) => {
                console.log("🔍 Name field changed:", e.target.value);
                setForm({ ...form, name: e.target.value });
              }}
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
          <DialogFooter className="flex justify-between">
            <Button
              type="button"
              variant="destructive"
              className="flex items-center gap-2"
              onClick={handleDelete}
              disabled={isDeleting}
            >
              <Trash className="h-4 w-4" />
              {isDeleting ? "Menghapus..." : "Hapus"}
            </Button>
            <div className="flex gap-2">
              <DialogClose asChild>
                <Button variant="outline">Batal</Button>
              </DialogClose>
              <Button type="submit" disabled={isUpdating}>
                {isUpdating ? "Menyimpan..." : "Simpan"}
              </Button>
            </div>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
