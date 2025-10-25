"use client";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Icons } from "@/app/dashboard/_components/icons";
import { refreshNasaPowerDataset } from "@/lib/fetch/files.fetch";

interface RefreshDatasetDialogProps {
  datasetId: string;
  datasetName: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function RefreshSingleDatasetDialog({
  datasetId,
  datasetName,
  open,
  onOpenChange,
}: RefreshDatasetDialogProps) {
  const { mutate, isPending, data, isSuccess, reset } = useMutation({
    mutationFn: () => refreshNasaPowerDataset(datasetId),
    onSuccess: (result) => {
      toast.success(`Dataset "${datasetName}" berhasil diperbarui`);
      onOpenChange(false);
      setTimeout(() => reset(), 300);
    },
    onError: (error: any) => {
      toast.error(
        error?.response?.data?.message || "Gagal memperbarui dataset"
      );
    },
  });

  const handleRefresh = () => mutate();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            <Icons.refresh className="h-5 w-5 inline mr-2" />
            Refresh Dataset
          </DialogTitle>
          <DialogDescription>
            Apakah Anda ingin memperbarui data <b>{datasetName}</b>?
          </DialogDescription>
        </DialogHeader>
        {!isPending && !isSuccess && (
          <div className="flex gap-2 mt-4">
            <Button
              onClick={() => onOpenChange(false)}
              variant="outline"
              className="flex-1"
            >
              Batal
            </Button>
            <Button
              onClick={handleRefresh}
              className="flex-1 bg-blue-600 hover:bg-blue-700"
            >
              <Icons.refresh className="mr-2 h-4 w-4" />
              Mulai Refresh
            </Button>
          </div>
        )}
        {isPending && (
          <div className="flex flex-col items-center justify-center py-8">
            <Icons.refresh className="h-12 w-12 animate-spin text-blue-600" />
            <p className="mt-4 text-center text-sm font-medium text-gray-700">
              Sedang memperbarui data...
            </p>
          </div>
        )}
        {isSuccess && data && (
          <div className="mt-4">
            <p className="text-green-700 font-semibold">{data.message}</p>
            <Button onClick={() => onOpenChange(false)} className="w-full mt-4">
              Tutup
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
