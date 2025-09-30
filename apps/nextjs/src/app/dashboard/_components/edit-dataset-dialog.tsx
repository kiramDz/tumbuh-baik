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
import { Icons } from "@/app/dashboard/_components/icons";
import { ConfirmationDeleteModal } from "@/components/ui/modal/confirmation-delete-modal";
import { ConfirmationUpdateModal } from "@/components/ui/modal/confirmation-update-modal";
import { toast } from "sonner";

interface EditDatasetDialogProps {
  dataset: {
    _id: string;
    name: string;
    source: string;
    collectionName: string;
    description?: string;
    status: string;
  };
}

export default function EditDatasetDialog({ dataset }: EditDatasetDialogProps) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
  const [isUpdateConfirmOpen, setIsUpdateConfirmOpen] = useState(false);
  const [form, setForm] = useState({
    name: dataset?.name || "",
    source: dataset?.source || "",
    collectionName: dataset?.collectionName || "",
    description: dataset?.description || "",
    status: dataset?.status || "raw",
  });
  // Mutation update
  const { mutate: updateDataset, isPending: isPending } = useMutation({
    mutationKey: ["update-dataset", dataset._id],
    mutationFn: (data: typeof form) => {
      if (!dataset?._id) {
        throw new Error("Dataset ID is missing");
      }

      return UpdateDatasetMeta(dataset._id, data);
    },
    onSuccess: (result) => {
      toast.success("Dataset berhasil diperbarui");
      queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
      setOpen(false);
      setIsUpdateConfirmOpen(false);
    },
    onError: (error: any) => {
      console.error("❌ Update failed, error:", error);
      console.error("❌ Error response:", error.response?.data);
      const errorMessage =
        error?.response?.data?.message || "Gagal memperbarui dataset";
      toast.error(errorMessage);
      setIsUpdateConfirmOpen(false);
    },
  });

  // Mutation delete
  const { mutate: deleteDataset, isPending: isDeleting } = useMutation({
    mutationKey: ["delete-dataset", dataset.collectionName], // Update key juga
    mutationFn: () => {
      if (!dataset?.collectionName) {
        throw new Error("Collection name is missing");
      }
      return DeleteDatasetMeta(dataset.collectionName);
    },
    onSuccess: () => {
      toast.success("Dataset berhasil dihapus");
      queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
      setOpen(false);
      setIsDeleteConfirmOpen(false);
    },
    onError: (error: any) => {
      console.error("Delete error:", error);
      const errorMessage =
        error?.response?.data?.message || "Gagal menghapus dataset";
      toast.error(errorMessage);
      setIsDeleteConfirmOpen(false);
    },
  });
  const handleDeleteClick = () => {
    setIsDeleteConfirmOpen(true);
  };

  const handleConfirmDelete = () => {
    deleteDataset();
  };

  const handleSubmitClick = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Submit button clicked, opening confirmation modal");
    setIsUpdateConfirmOpen(true);
  };

  const handleConfirmUpdate = () => {
    updateDataset(form);
  };

  return (
    <>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <Button
            className="group/menu flex h-8 w-8 items-center justify-center rounded-full bg-gray-100 transition-all duration-200 hover:bg-black hover:scale-110"
            onClick={(e) => e.stopPropagation()}
          >
            <Menu className="h-5 w-5 text-gray-600 transition-colors duration-200 group-hover/menu:text-white" />
          </Button>
        </DialogTrigger>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Edit Dataset</DialogTitle>
            <DialogDescription>
              Ubah metadata dataset atau hapus dataset.
            </DialogDescription>
          </DialogHeader>

          {/* Main form */}
          <form onSubmit={handleSubmitClick} className="space-y-4">
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

            {/* Action buttons */}
            <div className="flex justify-between pt-4">
              <Button
                type="button"
                variant="destructive"
                onClick={handleDeleteClick}
                disabled={isDeleting}
                className="flex items-center gap-2"
              >
                <Trash className="h-4 w-4" />
                {isDeleting ? "Menghapus..." : "Hapus"}
              </Button>

              <div className="flex gap-2">
                <DialogClose asChild>
                  <Button type="button" variant="outline">
                    Batal
                  </Button>
                </DialogClose>
                <Button
                  type="button" // Change to type="button"
                  disabled={isPending}
                  className="flex items-center gap-2"
                  onClick={(e) => {
                    e.preventDefault();
                    setIsUpdateConfirmOpen(true);
                  }}
                >
                  <Icons.save className="h-4 w-4" />
                  {isPending ? "Menyimpan..." : "Simpan"}
                </Button>
              </div>
            </div>
          </form>
        </DialogContent>
      </Dialog>
      {/* Delete Confirmation Modal */}
      <ConfirmationDeleteModal
        isOpen={isDeleteConfirmOpen}
        setIsOpen={setIsDeleteConfirmOpen}
        onConfirm={handleConfirmDelete}
        datasetName={dataset.name}
        collectionName={dataset.collectionName}
        isDeleting={isDeleting}
      />

      {/* Update Confirmation Modal */}
      <ConfirmationUpdateModal
        isOpen={isUpdateConfirmOpen}
        setIsOpen={setIsUpdateConfirmOpen}
        onConfirm={handleConfirmUpdate}
        datasetName={form.name}
        isUpdating={isPending}
      />
    </>
  );
}
