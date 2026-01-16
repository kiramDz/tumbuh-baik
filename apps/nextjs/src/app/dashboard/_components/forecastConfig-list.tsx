"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription } from "@/components/ui/dialog";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getForecastConfigs, updateForecastConfig, deleteForecastConfig } from "@/lib/fetch/files.fetch";
import { CheckCircle2, AlertTriangle, Clock, Loader2, Database, Pencil, Trash2, Calendar as CalendarIcon, Info, X, ChevronRight } from "lucide-react";
import { formatDistanceToNow, format } from "date-fns";
import { id } from "date-fns/locale";
import { useState } from "react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import axios from "axios";

interface ForecastConfig {
  _id: string;
  name: string;
  status: "done" | "failed" | "pending" | "running";
  columns?: { collectionName: string; columnName: string }[];
  errorMessage?: string;
  createdAt?: string;
  startDate?: string;
}

interface DatasetMeta {
  collectionName: string;
  name?: string;
  columns?: string[];
}

// Mapping nama kolom
const COLUMN_NAME_MAPPING: Record<string, string> = {
  // Kelembapan
  'RH_AVG_preprocessed': 'Kelembaban Rata-rata',
  'RH_AVG': "Kelembaban Rata-rata",

  // Suhu BMKG
  'TN': 'Suhu Minimum',
  'TX': 'Suhu Maksimum',
  'TAVG': 'Suhu Rata-rata',

  // Rainfall
  'RR_original': 'Curah Hujan Asli',
  'RR_imputed': 'Curah Hujan (Diperbaiki)',
  'is_outlier': 'Status Outlier',
  'RR_log': 'Curah Hujan (Log)',
  'RR_sqrt': 'Curah Hujan (Akar)',
  'RR_boxcox': 'Curah Hujan (Box-Cox)',
  'RR': 'Curah Hujan',

  // NASA
  'T2M': 'Suhu Udara',
  'T2M_MAX': 'Suhu Maksimum',
  'T2M_MIN': 'Suhu Minimum',
  'RH2M': 'Kelembaban Udara',
  'PRECTOTCORR': 'Curah Hujan',
  'ALLSKY_SFC_SW_DWN': 'Radiasi Matahari',
  'WS10M': 'Kecepatan Angin',
  'WS10M_MAX': 'Angin Maksimum',
  'WD10M': 'Arah Angin',

  'SS': 'Lamanya Penyinaran Matahari (Jam)',
  'FF_X': 'Kecepatan Angin Maksimum',
  'DDD_X': 'Arah Angin Maksimum',
  'FF_AVG': 'Kecepatan Angin Rata-rata',
  'DDD_CAR': 'Arah Angin Terbanyak',

  // Umum
  'Date': 'Tanggal',
  'Year': 'Tahun',
  'Month': 'Bulan',
  'Day': 'Hari',
  'month': 'Bulan',
  'day': 'Hari',
};

// Fungsi helper
const getDisplayName = (col: string): string => {
  if (COLUMN_NAME_MAPPING[col]) {
    return COLUMN_NAME_MAPPING[col];
  }
  return col
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
};

export function ForecastConfigList() {
  const { data = [], isLoading } = useQuery<ForecastConfig[]>({
    queryKey: ["forecast-config"],
    queryFn: getForecastConfigs,
  });

  if (isLoading) {
    return (
      <div className="flex gap-4 overflow-x-auto pb-2">
        {[...Array(3)].map((_, i) => (
          <Card key={i} className="w-[280px] shrink-0">
            <CardContent className="p-4 space-y-3">
              <Skeleton className="h-5 w-3/4" />
              <Skeleton className="h-4 w-1/3" />
              <div className="flex gap-2">
                <Skeleton className="h-5 w-16 rounded-full" />
                <Skeleton className="h-5 w-16 rounded-full" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center justify-center py-12">
          <Database className="h-10 w-10 text-muted-foreground mb-3" />
          <p className="font-medium text-base">Belum ada konfigurasi</p>
          <p className="text-sm text-muted-foreground">
            Buat konfigurasi baru untuk memulai
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <TooltipProvider>
      <ScrollArea className="w-full">
        <div className="flex gap-4 pb-4">
          {data.map((item) => (
            <ConfigCard key={item._id} config={item} />
          ))}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </TooltipProvider>
  );
}

function ConfigCard({ config }: { config: ForecastConfig }) {
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [editName, setEditName] = useState(config.name);
  const [editStartDate, setEditStartDate] = useState<Date | undefined>(
    config.startDate ? new Date(config.startDate) : config.createdAt ? new Date(config.createdAt) : undefined
  );
  const [editSelectedColumns, setEditSelectedColumns] = useState<{ collectionName: string; columnName: string }[]>(
    config.columns || []
  );
  
  const queryClient = useQueryClient();
  const status = getStatus(config.status);
  const StatusIcon = status.icon;

  // Query untuk dataset list
  const { data: datasetList = [], isLoading: isLoadingDatasets } = useQuery<DatasetMeta[]>({
    queryKey: ["dataset-meta"],
    queryFn: async () => {
      const res = await axios.get("/api/v1/dataset-meta");
      return res.data?.data || [];
    },
    enabled: isEditOpen, // Only fetch when edit dialog is open
  });

  // Mutation untuk update
  const updateMutation = useMutation({
    mutationFn: (data: { name: string; columns: { collectionName: string; columnName: string }[]; startDate: string }) =>
      updateForecastConfig(config._id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["forecast-config"] });
      setIsEditOpen(false);
      toast.success("Konfigurasi berhasil diperbarui");
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || "Gagal memperbarui konfigurasi");
    },
  });

  // Mutation untuk delete
  const deleteMutation = useMutation({
    mutationFn: () => deleteForecastConfig(config._id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["forecast-config"] });
      setIsDeleteOpen(false);
      toast.success("Konfigurasi berhasil dihapus");
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.message || "Gagal menghapus konfigurasi");
    },
  });

  const handleCheckbox = (collectionName: string, columnName: string, checked: boolean) => {
    if (checked) {
      setEditSelectedColumns((prev) => [...prev, { collectionName, columnName }]);
    } else {
      setEditSelectedColumns((prev) => prev.filter((item) => !(item.collectionName === collectionName && item.columnName === columnName)));
    }
  };

  const removeSelected = (collectionName: string, columnName: string) => {
    setEditSelectedColumns((prev) => prev.filter((item) => !(item.collectionName === collectionName && item.columnName === columnName)));
  };

  const handleUpdate = () => {
    if (!editName.trim()) {
      toast.error("Nama tidak boleh kosong");
      return;
    }

    if (!editStartDate) {
      toast.error("Tanggal mulai peramalan wajib diisi");
      return;
    }

    if (editSelectedColumns.length === 0) {
      toast.error("Pilih minimal satu kolom");
      return;
    }
    
    updateMutation.mutate({
      name: editName,
      columns: editSelectedColumns,
      startDate: format(editStartDate, "yyyy-MM-dd"),
    });
  };

  const handleDelete = () => {
    deleteMutation.mutate();
  };

  const handleOpenEditDialog = () => {
    setEditName(config.name);
    setEditStartDate(config.startDate ? new Date(config.startDate) : config.createdAt ? new Date(config.createdAt) : undefined);
    setEditSelectedColumns(config.columns || []);
    setIsEditOpen(true);
  };

  const endDate = editStartDate ? new Date(editStartDate.getFullYear() + 1, editStartDate.getMonth(), editStartDate.getDate()) : null;

  return (
    <>
      <Card className="w-[280px] shrink-0 hover:shadow-md transition-shadow">
        <CardContent className="p-4">
          {/* Header */}
          <div className="mb-3">
            <div className="flex items-start justify-between mb-1">
              <h4 className="font-medium text-base truncate flex-1">{config.name}</h4>
              <div className="flex gap-1 ml-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={handleOpenEditDialog}
                      disabled={config.status === "running" || config.status === "done"}
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </Button>
                  </TooltipTrigger>
                  {(config.status === "running" || config.status === "done") && (
                    <TooltipContent>
                      <p className="text-xs">
                        {config.status === "done" 
                          ? "Konfigurasi yang sudah selesai tidak dapat diedit" 
                          : "Tidak dapat diedit saat sedang berjalan"}
                      </p>
                    </TooltipContent>
                  )}
                </Tooltip>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 text-destructive hover:text-destructive"
                  onClick={() => setIsDeleteOpen(true)}
                  disabled={config.status === "running"}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
            <div className="flex items-center gap-1.5">
              <StatusIcon className={`h-4 w-4 ${status.color}`} />
              <span className={`text-sm ${status.color}`}>{status.label}</span>
              {config.createdAt && (
                <>
                  <span className="text-muted-foreground">·</span>
                  <span className="text-sm text-muted-foreground">
                    {formatDistanceToNow(new Date(config.createdAt), { addSuffix: true, locale: id })}
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Columns */}
          <div className="flex flex-wrap gap-1.5 mb-3">
            {config.columns?.slice(0, 3).map((col, idx) => (
              <Tooltip key={idx}>
                <TooltipTrigger asChild>
                  <Badge variant="secondary" className="text-sm font-normal cursor-help">
                    {getDisplayName(col.columnName)}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p className="text-xs text-white">
                    Kolom: <code className="bg-muted px-1 rounded text-foreground">{col.columnName}</code>
                  </p>
                </TooltipContent>
              </Tooltip>
            ))}
            {(config.columns?.length || 0) > 3 && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="outline" className="text-sm font-normal cursor-help">
                    +{(config.columns?.length || 0) - 3}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-[200px]">
                  <div className="text-xs space-y-1">
                    {config.columns?.slice(3).map((col, idx) => (
                      <p key={idx}>
                        {getDisplayName(col.columnName)}
                        <span className="text-muted-foreground ml-1">({col.columnName})</span>
                      </p>
                    ))}
                  </div>
                </TooltipContent>
              </Tooltip>
            )}
          </div>

          {/* Status Messages */}
          {config.status === "running" && (
            <div className="flex items-center gap-2 text-blue-600 text-sm">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Memproses...</span>
            </div>
          )}

          {config.errorMessage && (
            <p className="text-sm text-destructive line-clamp-1">
              {config.errorMessage}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Edit Dialog */}
      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent className="max-w-2xl max-h-[85vh] flex flex-col p-0">
          <DialogHeader className="px-6 pt-6 pb-4">
            <DialogTitle>Edit Konfigurasi</DialogTitle>
            <DialogDescription>
              Ubah pengaturan peramalan Holt-Winters yang sudah ada.
            </DialogDescription>
          </DialogHeader>

          <div className="flex-1 overflow-hidden px-6">
            <div className="space-y-5">
              {/* Nama Konfigurasi */}
              <div className="space-y-2">
                <Label htmlFor="edit-name" className="text-sm font-medium">
                  Nama Konfigurasi
                </Label>
                <Input
                  id="edit-name"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  placeholder="Nama konfigurasi"
                  className="h-10"
                />
              </div>

              {/* Tanggal Mulai Peramalan */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label className="text-sm font-medium">Tanggal Mulai Peramalan</Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="h-4 w-4 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-xs">
                      <p>Peramalan akan dilakukan selama 1 tahun dari tanggal yang dipilih.</p>
                    </TooltipContent>
                  </Tooltip>
                </div>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button 
                      variant="outline" 
                      className={cn(
                        "w-full justify-start text-left font-normal h-10", 
                        !editStartDate && "text-muted-foreground"
                      )}
                    >
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {editStartDate ? format(editStartDate, "PPP", { locale: id }) : "Pilih tanggal mulai"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0" align="start">
                    <Calendar 
                      mode="single" 
                      selected={editStartDate} 
                      onSelect={setEditStartDate} 
                      initialFocus 
                      locale={id} 
                    />
                  </PopoverContent>
                </Popover>

                {/* Preview Tanggal Akhir */}
                {endDate && (
                  <p className="text-xs text-muted-foreground">
                    Peramalan hingga: <span className="font-medium text-foreground">{format(endDate, "PPP", { locale: id })}</span>
                  </p>
                )}
              </div>

              <Separator />

              {/* Dataset Selection */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Label className="text-sm font-medium">Pilih Kolom Dataset</Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="h-4 w-4 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-xs">
                      <p>Pilih kolom data yang ingin Anda ramalkan. Hover pada nama kolom untuk melihat nama teknis aslinya.</p>
                    </TooltipContent>
                  </Tooltip>
                </div>
                
                {isLoadingDatasets ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : datasetList.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <Database className="h-12 w-12 text-muted-foreground mb-3" />
                    <p className="text-sm text-muted-foreground">Tidak ada dataset tersedia</p>
                  </div>
                ) : (
                  <ScrollArea className="h-[200px]">
                    <div className="space-y-3 pr-4">
                      {datasetList.map((dataset) => (
                        <Card key={dataset.collectionName} className="border-muted">
                          <CardHeader className="py-3 px-4">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Database className="h-4 w-4 text-primary" />
                                <CardTitle className="text-sm font-medium">
                                  {dataset.name || dataset.collectionName}
                                </CardTitle>
                              </div>
                              <Badge variant="outline" className="text-xs font-normal">
                                {dataset.columns?.length || 0} kolom
                              </Badge>
                            </div>
                          </CardHeader>
                          <CardContent className="pt-0 pb-3 px-4">
                            <div className="grid grid-cols-2 gap-1.5">
                              {dataset.columns?.map((col: string) => {
                                const isChecked = editSelectedColumns.some(
                                  (item) => item.collectionName === dataset.collectionName && item.columnName === col
                                );
                                const displayName = getDisplayName(col);

                                return (
                                  <Tooltip key={col}>
                                    <TooltipTrigger asChild>
                                      <label 
                                        className={`
                                          flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer
                                          text-sm transition-all duration-150
                                          ${isChecked 
                                            ? 'bg-primary/10 text-primary border border-primary/30' 
                                            : 'hover:bg-muted border border-transparent'
                                          }
                                        `}
                                      >
                                        <Checkbox 
                                          checked={isChecked} 
                                          onCheckedChange={(checked) => 
                                            handleCheckbox(dataset.collectionName, col, !!checked)
                                          }
                                          className="data-[state=checked]:bg-primary"
                                        />
                                        <span className="truncate flex-1">
                                          {displayName}
                                        </span>
                                      </label>
                                    </TooltipTrigger>
                                    <TooltipContent side="top">
                                      <p className="text-xs text-white">
                                        Kolom: <code className="bg-muted px-1 rounded text-foreground">{col}</code>
                                      </p>
                                    </TooltipContent>
                                  </Tooltip>
                                );
                              })}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </div>

              {/* Selected Columns Preview */}
              {editSelectedColumns.length > 0 && (
                <div className="space-y-2 pb-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium">Kolom Terpilih</Label>
                    <Badge variant="secondary" className="text-xs">
                      {editSelectedColumns.length} dipilih
                    </Badge>
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-[80px] overflow-y-auto">
                    {editSelectedColumns.map((item, index) => (
                      <Badge 
                        key={index} 
                        variant="secondary"
                        className="pl-2 pr-1 py-1 gap-1 text-sm"
                      >
                        <span className="text-muted-foreground max-w-[80px] truncate">
                          {item.collectionName}
                        </span>
                        <ChevronRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                        <span className="font-medium">
                          {getDisplayName(item.columnName)}
                        </span>
                        <button
                          type="button"
                          onClick={() => removeSelected(item.collectionName, item.columnName)}
                          className="ml-1 rounded-full p-0.5 hover:bg-destructive/20 hover:text-destructive transition-colors"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <DialogFooter className="px-6 py-4 border-t bg-muted/30">
            <Button variant="outline" onClick={() => setIsEditOpen(false)} disabled={updateMutation.isPending}>
              Batal
            </Button>
            <Button 
              onClick={handleUpdate} 
              disabled={updateMutation.isPending || !editName.trim() || editSelectedColumns.length === 0 || !editStartDate}
            >
              {updateMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Menyimpan...
                </>
              ) : (
                "Simpan Perubahan"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={isDeleteOpen} onOpenChange={setIsDeleteOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Hapus Konfigurasi?</AlertDialogTitle>
            <AlertDialogDescription>
              Apakah Anda yakin ingin menghapus konfigurasi <strong>{config.name}</strong>?
              Tindakan ini tidak dapat dibatalkan.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Batal</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              disabled={deleteMutation.isPending}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleteMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Menghapus...
                </>
              ) : (
                "Hapus"
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

function getStatus(status: string) {
  const config: Record<string, { label: string; icon: typeof CheckCircle2; color: string }> = {
    done: { label: "Selesai", icon: CheckCircle2, color: "text-green-600" },
    failed: { label: "Gagal", icon: AlertTriangle, color: "text-destructive" },
    pending: { label: "Menunggu", icon: Clock, color: "text-muted-foreground" },
    running: { label: "Berjalan", icon: Loader2, color: "text-blue-600" },
  };
  return config[status] || config.pending;
}
