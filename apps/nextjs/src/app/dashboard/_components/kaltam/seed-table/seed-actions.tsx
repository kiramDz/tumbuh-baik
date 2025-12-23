"use client";

import { useState } from "react";
import { EllipsisVertical, Pencil, Trash2 } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { updateSeed, deleteSeed } from "@/lib/fetch/files.fetch";
import { SeedType } from "@/types/table-schema";

interface SeedActionsProps {
  seed: SeedType;
}

export function SeedActions({ seed }: SeedActionsProps) {
  const queryClient = useQueryClient();
  const [popoverOpen, setPopoverOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [name, setName] = useState(seed.name);
  const [duration, setDuration] = useState(seed.duration.toString());

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: (data: { name: string; duration: number }) => updateSeed(typeof seed._id === "string" ? seed._id : seed._id.$oid, data),
    onSuccess: () => {
      toast.success("Berhasil memperbarui bibit");
      queryClient.invalidateQueries({ queryKey: ["get-seed"] });
      setEditDialogOpen(false);
      setPopoverOpen(false);
    },
    onError: () => {
      toast.error("Gagal memperbarui bibit");
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: () => deleteSeed(typeof seed._id === "string" ? seed._id : seed._id.$oid),
    onSuccess: () => {
      toast.success("Berhasil menghapus bibit");
      queryClient.invalidateQueries({ queryKey: ["get-seed"] });
      setPopoverOpen(false);
    },
    onError: () => {
      toast.error("Gagal menghapus bibit");
    },
  });

  const handleEdit = () => {
    setPopoverOpen(false);
    setEditDialogOpen(true);
  };

  const handleDelete = () => {
    setPopoverOpen(false);
    deleteMutation.mutate(); 
  };

  const handleSubmitEdit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!name.trim()) return toast.error("Nama bibit tidak boleh kosong");
    if (!duration || Number(duration) <= 0) return toast.error("Durasi harus lebih dari 0 hari");

    updateMutation.mutate({ name: name.trim(), duration: Number(duration) });
  };

  return (
    <>
      <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
        <PopoverTrigger asChild>
          <Button variant="ghost" size="icon">
            <EllipsisVertical className="h-4 w-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-40 p-2" align="end">
          <div className="flex flex-col gap-1">
            <Button variant="ghost" size="sm" className="justify-start gap-2" onClick={handleEdit}>
              <Pencil className="h-4 w-4" />
              Edit
            </Button>
            <Button variant="ghost" size="sm" className="justify-start gap-2 text-destructive hover:text-destructive" onClick={handleDelete} disabled={deleteMutation.isPending}>
              <Trash2 className="h-4 w-4" />
              {deleteMutation.isPending ? "Menghapus..." : "Hapus"}
            </Button>
          </div>
        </PopoverContent>
      </Popover>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Edit Bibit</DialogTitle>
            <DialogDescription>Perbarui data bibit padi di bawah ini.</DialogDescription>
          </DialogHeader>
          <form onSubmit={handleSubmitEdit}>
            <div className="grid gap-4">
              <div className="grid gap-3">
                <Label htmlFor="edit-name">Nama Bibit</Label>
                <Input id="edit-name" value={name} placeholder="Masukkan nama bibit" onChange={(e) => setName(e.target.value)} />
              </div>
              <div className="grid gap-3">
                <Label htmlFor="edit-duration">Durasi (hari)</Label>
                <Input id="edit-duration" type="number" value={duration} placeholder="Masukkan durasi dalam hari" onChange={(e) => setDuration(e.target.value)} />
              </div>
            </div>
            <DialogFooter className="mt-4">
              <Button type="button" variant="outline" onClick={() => setEditDialogOpen(false)}>
                Batal
              </Button>
              <Button type="submit" disabled={updateMutation.isPending}>
                {updateMutation.isPending ? "Menyimpan..." : "Simpan"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}
