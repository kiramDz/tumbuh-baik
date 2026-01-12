"use client";

import { createSeed } from "@/lib/fetch/files.fetch";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogClose } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";

export default function AddSeedDialog() {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [duration, setDuration] = useState("");
  const [open, setOpen] = useState(false);

  const { mutate, isPending } = useMutation({
    mutationKey: ["add-seed"],
    mutationFn: createSeed,
    onSuccess: () => {
      toast.success("Berhasil menambahkan bibit");
      queryClient.invalidateQueries({ queryKey: ["seeds"] });
      setOpen(false);
      setName("");
      setDuration("");
    },
    onError: () => {
      toast.error("Gagal menambahkan bibit");
    },
  });

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!name.trim()) return toast.error("Nama bibit tidak boleh kosong");
    if (!duration || Number(duration) <= 0) return toast.error("Durasi harus lebih dari 0 hari");

    mutate({ name: name.trim(), duration: Number(duration) });
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild className="max-w-[10rem]">
        <Button variant="outline">Tambah Data</Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Tambah Bibit</DialogTitle>
          <DialogDescription>Masukkan data bibit padi di bawah ini.</DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4">
            <div className="grid gap-3">
              <Label htmlFor="name">Nama Bibit</Label>
              <Input id="name" value={name} placeholder="Masukkan nama bibit" onChange={(e) => setName(e.target.value)} />
            </div>
            <div className="grid gap-3">
              <Label htmlFor="duration">Durasi (hari)</Label>
              <Input id="duration" type="number" value={duration} placeholder="Masukkan durasi dalam hari" onChange={(e) => setDuration(e.target.value)} />
            </div>
          </div>
          <DialogFooter className="mt-2">
            <DialogClose asChild>
              <Button variant="outline">Batal</Button>
            </DialogClose>
            <Button type="submit" disabled={isPending} onClick={() => console.log("SUBMIT BUTTON CLICKED")}>
              {isPending ? "Menyimpan..." : "Simpan"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
