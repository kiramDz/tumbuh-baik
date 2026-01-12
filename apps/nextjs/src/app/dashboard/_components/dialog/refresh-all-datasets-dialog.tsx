"use client";
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { Button } from "@/components/ui/button";
import { Icons } from "@/app/dashboard/_components/icons";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { refreshAllNasaDatasets, RefreshResult } from "@/lib/fetch/files.fetch";

export default function RefreshAllDatasetsDialog() {
  const [open, setOpen] = useState(false);
  const queryClient = useQueryClient();

  // Mutation to refresh all datasets
  const { mutate, isPending, data, isSuccess, reset } = useMutation({
    mutationFn: refreshAllNasaDatasets,
    onSuccess: (result) => {
      // Handle case when no datasets found
      if (!result.data) {
        toast(
          result.message || "Tidak ada dataset NASA POWER yang perlu diperbarui"
        );
        // Close dialog and reset state
        setOpen(false);
        setTimeout(() => {
          reset();
        }, 300);
        return;
      }

      // Show appropriate toast based on result
      if (result.data.failed > 0) {
        toast(
          (t) => (
            <div className="flex items-center gap-3">
              <span>
                Pembaruan selesai dengan beberapa kesalahan:{" "}
                {result.data.refreshed} berhasil, {result.data.failed} gagal
              </span>
              <button
                onClick={() => toast.dismiss(t.id)}
                className="px-2 py-1 bg-gray-800 text-white rounded text-sm font-medium hover:bg-gray-700"
              >
                Dismiss
              </button>
            </div>
          ),
          {
            icon: "⚠️",
            duration: 6000,
          }
        );
      } else if (result.data.refreshed === 0) {
        toast.success(
          `Semua dataset sudah up-to-date: ${result.data.alreadyUpToDate} dataset tidak memerlukan pembaruan`
        );
      } else {
        toast.success(
          `Pembaruan berhasil! ${result.data.refreshed} dataset berhasil diperbarui`
        );
      }

      // Invalidate queries to refresh Data
      queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
    },
    onError: (error: any) => {
      console.error("Refresh all failed:", error);
      toast.error(
        error?.response?.data?.message ||
          error?.message ||
          "Gagal memperbarui dataset: Terjadi kesalahan tidak terduga"
      );
    },
  });
  const handleStartRefresh = () => {
    mutate();
  };
  const handleClose = () => {
    setOpen(false);
    setTimeout(() => {
      reset();
    }, 300);
  };
  const getStatusBadge = (result: RefreshResult) => {
    if (result.refreshResult === "success") {
      return (
        <Badge variant="default" className="bg-green-500">
          Success
        </Badge>
      );
    }
    if (result.refreshResult === "no-new-data") {
      return <Badge variant="secondary">Up-to-date</Badge>;
    }
    if (result.refreshResult === "failed") {
      return <Badge variant="destructive">Failed</Badge>;
    }
    return <Badge variant="outline">{result.refreshResult}</Badge>;
  };
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          className="border-blue-600 text-blue-700 hover:bg-blue-50 hover:border-blue-700 font-semibold flex items-center gap-2 shadow-sm transition-colors"
        >
          <Icons.refresh className="h-4 w-4" />
          <span className="hidden sm:inline">Refresh All</span>
        </Button>
      </DialogTrigger>

      <DialogContent className="sm:max-w-[600px] max-h-[80vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Icons.refresh className="h-5 w-5" />
            Perbarui Semua Dataset NASA POWER?
          </DialogTitle>
          <DialogDescription>
            Mengambil tanggal terbaru untuk semua dataset NASA POWER.
          </DialogDescription>
        </DialogHeader>

        {/* Initial State - Confirmation */}
        {!isPending && !isSuccess && (
          <div className="flex flex-col gap-4 py-4">
            <div className="rounded-lg border bg-blue-50 p-4">
              <div className="flex gap-3">
                <Icons.info className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-blue-800">
                  <p className="font-medium mb-1">Informasi:</p>
                  <ul className="list-disc pl-4 space-y-1">
                    <li>Proses akan memeriksa semua dataset NASA POWER</li>
                    <li>Hanya dataset yang menggunakan API akan di-refresh</li>
                    <li>Proses mungkin membutuhkan waktu beberapa menit</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={handleClose}
                variant="outline"
                className="flex-1"
              >
                Batal
              </Button>
              <Button
                onClick={handleStartRefresh}
                className="flex-1 bg-blue-600 hover:bg-blue-700"
              >
                <Icons.refresh className="mr-2 h-4 w-4" />
                Mulai Pembaruan
              </Button>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isPending && (
          <div className="flex flex-col items-center justify-center py-8">
            <div className="relative">
              <Icons.refresh className="h-12 w-12 animate-spin text-blue-600" />
              <div className="absolute inset-0 rounded-full border-4 border-blue-200 border-t-transparent animate-spin" />
            </div>
            <p className="mt-4 text-center text-sm font-medium text-gray-700">
              Sedang memperbarui data NASA POWER...
            </p>
            <p className="mt-1 text-center text-xs text-gray-500">
              Mohon tunggu, proses sedang berlangsung
            </p>
          </div>
        )}

        {/* Success State with Details */}
        {isSuccess && data && data.data && (
          <div className="space-y-4 py-2">
            {/* Summary Stats */}
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border bg-gray-50 p-3">
                <div className="text-2xl font-bold text-gray-900">
                  {data.data.total}
                </div>
                <div className="text-xs text-gray-600">Total Dataset</div>
              </div>
              <div className="rounded-lg border bg-green-50 p-3">
                <div className="text-2xl font-bold text-green-600">
                  {data.data.refreshed}
                </div>
                <div className="text-xs text-green-700">
                  Berhasil Diperbarui
                </div>
              </div>
              <div className="rounded-lg border bg-blue-50 p-3">
                <div className="text-2xl font-bold text-blue-600">
                  {data.data.alreadyUpToDate}
                </div>
                <div className="text-xs text-blue-700">Sudah Terbaru</div>
              </div>
              <div className="rounded-lg border bg-red-50 p-3">
                <div className="text-2xl font-bold text-red-600">
                  {data.data.failed}
                </div>
                <div className="text-xs text-red-700">Gagal</div>
              </div>
            </div>

            {/* Detailed Results */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium text-gray-700">
                  Detail Pembaruan
                </h4>
                <span className="text-xs text-gray-500">
                  {data.data.details.length} dataset
                </span>
              </div>

              <ScrollArea className="h-[240px] rounded-md border">
                <div className="p-3 space-y-2">
                  {data.data.details.map((result) => (
                    <div
                      key={result.id}
                      className="flex items-start gap-3 rounded-lg border bg-white p-3 hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {result.name}
                          </p>
                          {getStatusBadge(result)}
                        </div>
                        <div className="space-y-0.5">
                          {result.newRecordsCount !== undefined &&
                            result.newRecordsCount > 0 && (
                              <p className="text-xs text-gray-600">
                                <Icons.plus className="inline h-3 w-3 mr-1" />
                                {result.newRecordsCount} data baru ditambahkan
                              </p>
                            )}
                          {result.lastRecord && (
                            <p className="text-xs text-gray-500">
                              <Icons.calendar className="inline h-3 w-3 mr-1" />
                              Terakhir:{" "}
                              {new Date(result.lastRecord).toLocaleDateString(
                                "id-ID",
                                {
                                  day: "numeric",
                                  month: "short",
                                  year: "numeric",
                                }
                              )}
                            </p>
                          )}
                          {result.reason && (
                            <p className="text-xs text-red-600">
                              <Icons.alertCircle className="inline h-3 w-3 mr-1" />
                              {result.reason}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
            {/* Close Button */}
            <Button onClick={handleClose} className="w-full">
              <Icons.check className="mr-2 h-4 w-4" />
              Tutup
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
