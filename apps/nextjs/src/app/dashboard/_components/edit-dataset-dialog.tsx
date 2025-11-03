"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import React, { useState } from "react";
import { useEffect } from "react";
import {
  UpdateDatasetMeta,
  DeleteDatasetMeta,
  getNasaPowerRefreshStatus,
  refreshNasaPowerDataset,
} from "@/lib/fetch/files.fetch";
import { Button } from "@/components/ui/button";
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
import { Icons } from "@/app/dashboard/_components/icons";
import { ConfirmationDeleteModal } from "@/components/ui/modal/confirmation-delete-modal";
import { ConfirmationUpdateModal } from "@/components/ui/modal/confirmation-update-modal";
import toast from "react-hot-toast";

interface EditDatasetDialogProps {
  dataset: {
    _id: string;
    name: string;
    source: string;
    collectionName: string;
    description?: string;
    status: string;
    isAPI?: boolean; // Add this
    apiConfig?: {
      type: string;
      params?: any;
    }; // Add this
    lastUpdated?: string;
  };
}

export default function EditDatasetDialog({ dataset }: EditDatasetDialogProps) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
  const [isUpdateConfirmOpen, setIsUpdateConfirmOpen] = useState(false);
  const [refreshStatus, setRefreshStatus] = useState({
    canRefresh: false,
    daysSinceLastRecord: 0,
    lastRecordDate: "",
    message: "",
    isLoading: false,
  });
  // Form state
  const [form, setForm] = useState({
    name: dataset?.name || "",
    source: dataset?.source || "",
    collectionName: dataset?.collectionName || "",
    description: dataset?.description || "",
    status: dataset?.status || "raw",
  });
  // useEffect for NASA latest date
  useEffect(() => {
    if (open && dataset.isAPI && dataset.apiConfig?.type === "nasa-power") {
      setRefreshStatus((prev) => ({ ...prev, isLoading: true }));

      getNasaPowerRefreshStatus(dataset._id)
        .then((status) => {
          setRefreshStatus({
            canRefresh: status.canRefresh,
            daysSinceLastRecord: status.daysSinceLastRecord || 0,
            lastRecordDate: status.lastRecordDate || "",
            message: status.message,
            isLoading: false,
          });
        })
        .catch((error) => {
          console.error("Error fetching refresh status:", error);
          // On error, allow refresh attempt
          setRefreshStatus({
            canRefresh: true,
            daysSinceLastRecord: 0,
            lastRecordDate: "",
            message: "Unable to check status",
            isLoading: false,
          });
        });
    }
  }, [open, dataset._id, dataset.isAPI, dataset.apiConfig?.type]);

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

  // Mutation refresh (for NASA POWER datasets)
  const { mutate: refreshDataset, isPending: isRefreshing } = useMutation({
    mutationKey: ["refresh-nasa-dataset", dataset._id],
    mutationFn: () => {
      if (!dataset?._id) {
        throw new Error("Dataset ID is missing");
      }
      return refreshNasaPowerDataset(dataset._id);
    },
    onSuccess: (data) => {
      // Check if there were records updated
      if (
        data.data?.newRecordsCount === 0 ||
        data.message?.includes("up to date") ||
        data.message?.includes("No new data")
      ) {
        toast(
          `Dataset sudah memiliki data terbaru\nData terakhir: ${new Date(
            data.data?.lastUpdated || dataset.lastUpdated || ""
          ).toLocaleDateString("id-ID")}`,
          {
            duration: 5000,
            icon: "ℹ️",
            position: "bottom-right",
          }
        );
      } else {
        toast.success(
          `Berhasil memperbarui ${
            data.data?.newRecordsCount || 0
          } data baru\nTotal data: ${
            data.data?.dataset?.totalRecords || "N/A"
          }`,
          {
            duration: 5000,
            position: "bottom-right",
          }
        );
      }

      // Update refresh status after successful refresh
      setRefreshStatus({
        canRefresh: false,
        daysSinceLastRecord: 0,
        lastRecordDate: new Date().toISOString(),
        message: "Dataset sudah up-to-date",
        isLoading: false,
      });

      queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
      // Don't close dialog, let user see the result
    },
    onError: (error: any) => {
      console.error("❌ Refresh failed, error:", error);

      // Special handling for "already up to date" messages
      if (
        error?.response?.data?.message?.includes("up to date") ||
        error?.message?.includes("up to date")
      ) {
        toast(
          `Dataset sudah memiliki data terbaru\nData terakhir: ${new Date(
            refreshStatus.lastRecordDate || dataset.lastUpdated || ""
          ).toLocaleDateString("id-ID")}`,
          {
            duration: 5000,
            icon: "ℹ️",
            position: "bottom-right",
          }
        );

        // Update refresh status
        setRefreshStatus({
          canRefresh: false,
          daysSinceLastRecord: 0,
          lastRecordDate: dataset.lastUpdated || "",
          message: "Dataset sudah up-to-date",
          isLoading: false,
        });
        return;
      }

      const errorMessage =
        error?.response?.data?.message || "Gagal memperbarui dataset";
      toast.error(errorMessage);
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
      toast.success(`Dataset "${dataset.name}" telah dihapus.`);
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

  // funtion handle refresh
  const handleRefreshClick = () => {
    // Prevent refresh if status check is still loading
    if (refreshStatus.isLoading) {
      toast("Memeriksa status data...", {
        icon: "ℹ️",
        position: "bottom-right",
      });
      return;
    }

    // Prevent refresh if data is already up-to-date
    if (!refreshStatus.canRefresh) {
      toast(
        `Dataset sudah memiliki data terbaru\nData terakhir: ${new Date(
          refreshStatus.lastRecordDate || dataset.lastUpdated || ""
        ).toLocaleDateString("id-ID")}`,
        {
          duration: 5000,
          icon: "ℹ️",
          position: "bottom-right",
        }
      );
      return;
    }

    refreshDataset();
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
            <Icons.menu className="h-5 w-5 text-gray-600 transition-colors duration-200 group-hover/menu:text-white" />
          </Button>
        </DialogTrigger>
        <DialogContent className="w-[95vw] max-w-[500px] max-h-[90vh] overflow-y-auto">
          {" "}
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
                //disabled={dataset.isAPI} // Optionally disable for API datasets
              >
                {/* Show status options based on dataset source/type */}
                {dataset.isAPI && dataset.apiConfig?.type === "nasa-power" ? (
                  <>
                    <option value="raw">Raw</option>
                    <option value="latest">Latest</option>
                    <option value="preprocessed">Preprocessed</option>
                    <option value="validated">Validated</option>
                    <option value="archived">Archived</option>
                  </>
                ) : (
                  <>
                    <option value="raw">Raw</option>
                    <option value="cleaned">Cleaned</option>
                    <option value="validated">Validated</option>
                    <option value="archived">Archived</option>
                  </>
                )}
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
            <div className="space-y-4 pt-4 border-t">
              {/* NASA POWER Refresh Section - Pindah ke atas */}
              {dataset.isAPI && dataset.apiConfig?.type === "nasa-power" && (
                <div className="flex flex-col gap-2 p-3 bg-gray-50 rounded-lg border">
                  <div className="text-sm">
                    <p className="font-medium text-gray-700 mb-2">
                      Status Data NASA POWER
                    </p>
                    {refreshStatus.isLoading ? (
                      <p className="text-xs text-blue-600 flex items-center gap-1">
                        <Icons.refresh className="h-3 w-3 animate-spin" />
                        Memeriksa status...
                      </p>
                    ) : (
                      <p className="text-xs">
                        {refreshStatus.canRefresh ? (
                          <span className="text-green-600 font-medium">
                            ✓ Tersedia {refreshStatus.daysSinceLastRecord} hari
                            data baru
                          </span>
                        ) : (
                          <span className="text-gray-500">
                            ✓ Up-to-date:{" "}
                            {new Date(
                              dataset.lastUpdated || ""
                            ).toLocaleDateString("id-ID", {
                              day: "numeric",
                              month: "short",
                              year: "numeric",
                            })}
                          </span>
                        )}
                      </p>
                    )}
                  </div>

                  <Button
                    type="button"
                    variant={refreshStatus.canRefresh ? "secondary" : "outline"}
                    onClick={handleRefreshClick}
                    disabled={
                      isRefreshing ||
                      !refreshStatus.canRefresh ||
                      refreshStatus.isLoading
                    }
                    className="flex items-center justify-center gap-2 w-full"
                    size="sm"
                  >
                    <Icons.refresh
                      className={`h-4 w-4 ${
                        isRefreshing ? "animate-spin" : ""
                      }`}
                    />
                    {isRefreshing
                      ? "Memperbarui data..."
                      : refreshStatus.canRefresh
                      ? `Refresh Data (${refreshStatus.daysSinceLastRecord} hari)`
                      : "Data Terbaru"}
                  </Button>
                </div>
              )}

              {/* Bottom Action Buttons */}
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
                {/* Delete Button - Left */}
                <Button
                  type="button"
                  variant="destructive"
                  onClick={handleDeleteClick}
                  disabled={isDeleting}
                  className="flex items-center justify-center gap-2 w-full sm:w-auto"
                  size="sm"
                >
                  <Icons.trash className="h-4 w-4" />
                  {isDeleting ? "Menghapus..." : "Hapus"}
                </Button>

                {/* Cancel & Save Buttons - Right */}
                <div className="flex items-center gap-2 w-full sm:w-auto">
                  <DialogClose asChild>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="flex-1 sm:flex-none"
                    >
                      Batal
                    </Button>
                  </DialogClose>
                  <Button
                    type="button"
                    disabled={isPending}
                    className="flex items-center justify-center gap-2 flex-1 sm:flex-none"
                    onClick={(e) => {
                      e.preventDefault();
                      setIsUpdateConfirmOpen(true);
                    }}
                    size="sm"
                  >
                    <Icons.save className="h-4 w-4" />
                    {isPending ? "Menyimpan..." : "Simpan"}
                  </Button>
                </div>
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
