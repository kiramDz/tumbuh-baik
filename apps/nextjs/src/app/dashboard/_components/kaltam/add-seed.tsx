"use client";

import { createSeed } from "@/lib/fetch/files.fetch";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
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
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { Plus, Loader2 } from "lucide-react";

export default function AddSeedDialog() {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [duration, setDuration] = useState("");
  const [open, setOpen] = useState(false);

  const { mutate, isPending } = useMutation({
    mutationKey: ["add-seed"],
    mutationFn: createSeed,
    onSuccess: () => {
      toast.success("Varietas berhasil ditambahkan");
      queryClient.invalidateQueries({ queryKey: ["seeds"] });
      setOpen(false);
      resetForm();
    },
    onError: () => {
      toast.error("Gagal menambahkan varietas");
    },
  });

  const resetForm = () => {
    setName("");
    setDuration("");
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!name.trim()) {
      return toast.error("Nama varietas wajib diisi");
    }

    if (!duration || Number(duration) <= 0) {
      return toast.error("Durasi harus lebih dari 0 hari");
    }

    mutate({ name: name.trim(), duration: Number(duration) });
  };

  const isFormValid = name.trim() && duration && Number(duration) > 0;

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <Plus className="w-4 h-4 mr-2" />
          Tambah Varietas
        </Button>
      </DialogTrigger>

      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Tambah Varietas Baru</DialogTitle>
          <DialogDescription>
            Masukkan data varietas padi untuk kalender tanam.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4 pt-2">
          <div className="space-y-2">
            <Label htmlFor="name">
              Nama Varietas <span className="text-destructive">*</span>
            </Label>
            <Input
              id="name"
              value={name}
              placeholder="Ciherang, IR64, Mekongga..."
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="duration">
              Durasi Tanam (hari) <span className="text-destructive">*</span>
            </Label>
            <Input
              id="duration"
              type="number"
              min="1"
              value={duration}
              placeholder="110"
              onChange={(e) => setDuration(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Jumlah hari dari tanam hingga panen
            </p>
          </div>

          <DialogFooter className="gap-2 sm:gap-0 pt-2">
            <DialogClose asChild>
              <Button type="button" variant="outline">
                Batal
              </Button>
            </DialogClose>
            <Button type="submit" disabled={isPending || !isFormValid}>
              {isPending ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Menyimpan...
                </>
              ) : (
                "Simpan"
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
