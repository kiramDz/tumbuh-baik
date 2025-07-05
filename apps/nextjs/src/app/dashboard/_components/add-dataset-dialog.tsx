"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { AddDatasetMeta } from "@/lib/fetch/files.fetch";
import { Button } from "@/components/ui/button";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogClose } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";

export default function AddDatasetDialog() {
  const queryClient = useQueryClient();
  const [form, setForm] = useState({
    name: "",
    source: "",
    filename: "",
    fileType: "csv",
    collectionTarget: "",
    month: "",
    timestamp: "",
    description: "",
  });

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
        filename: "",
        fileType: "csv",
        collectionTarget: "",
        month: "",
        timestamp: "",
        description: "",
      });
    },
    onError: () => {
      toast.error("Gagal menambahkan dataset");
    },
  });

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!form.name || !form.source || !form.filename || !form.collectionTarget || !form.month || !form.timestamp) {
      return toast.error("Mohon lengkapi semua data wajib");
    }
    mutate(form);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          onClick={() => {
            console.log("Button clicked!"); // Debug log
            setOpen(true);
          }}
        >
          Tambah Dataset
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Tambah Dataset</DialogTitle>
          <DialogDescription>Masukkan metadata dataset untuk upload dan penyimpanan.</DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="grid gap-4 py-4">
          {[
            { id: "name", label: "Nama Dataset", required: true },
            { id: "source", label: "Sumber", required: true },
            { id: "filename", label: "Nama File", required: true },
            { id: "collectionTarget", label: "Koleksi Tujuan", required: true },
            { id: "month", label: "Bulan (YYYY-MM)", required: true },
            { id: "timestamp", label: "Timestamp (ISO)", required: true },
            { id: "description", label: "Deskripsi", required: false },
          ].map((field) => (
            <div key={field.id} className="grid gap-2">
              <Label htmlFor={field.id}>{field.label}</Label>
              <Input id={field.id} value={form[field.id as keyof typeof form]} onChange={(e) => setForm({ ...form, [field.id]: e.target.value })} required={field.required} />
            </div>
          ))}
          <div className="grid gap-2">
            <Label htmlFor="fileType">Tipe File</Label>
            <select id="fileType" value={form.fileType} onChange={(e) => setForm({ ...form, fileType: e.target.value })} className="border rounded px-3 py-2">
              <option value="csv">CSV</option>
              <option value="json">JSON</option>
            </select>
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
