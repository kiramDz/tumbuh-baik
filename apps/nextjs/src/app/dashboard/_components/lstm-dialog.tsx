"use client"

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import axios from "axios";
import { Dialog, DialogContent, DialogTrigger, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { toast } from "sonner";
import { createLSTMConfig } from "@/lib/fetch/files.fetch";
import { Plus, Database, X, Loader2, ChevronRight, Info, Calendar as Calendar1 } from "lucide-react";
import { format } from "date-fns";
import { id } from "date-fns/locale";
import { cn } from "@/lib/utils";

interface LSTMDialogProps {
    onSubmit?: () => void;
}

interface DatasetMeta {
    collectionName: string;
    name?: string;
    columns?: string[];
}

// Mapping nama kolom - dipindahkan ke luar component agar tidak re-create setiap render
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

export function LSTMDialog({ onSubmit }: LSTMDialogProps) {
    const [open, setOpen] = useState(false);
    const [name, setName] = useState("");
    const [startDate, setStartDate] = useState<Date>();
    const [selected, setSelected] = useState<{ collectionName: string; columnName: string }[]>([]);

    const queryClient = useQueryClient();
    const { data: datasetList = [], isLoading: isLoadingDatasets } = useQuery<DatasetMeta[]>({
        queryKey: ["dataset-meta"],
        queryFn: async () => {
            const res = await axios.get("/api/v1/dataset-meta");
            return res.data?.data || [];
        },
    });

    const { mutate, isPending } = useMutation({
        mutationKey: ["lstm-config"],
        mutationFn: createLSTMConfig,
        onSuccess: () => {
            toast.success("Konfigurasi berhasil disimpan");
            queryClient.invalidateQueries({ queryKey: ["lstm-config"] });
            onSubmit?.();
            setOpen(false);
            setName("");
            setSelected([]);
            setStartDate(undefined);
        },
        onError: (error) => {
            console.error("Error saving LSTM config:", error);
            toast.error("Gagal menyimpan konfigurasi");
        },
    });

    const handleCheckbox = (collectionName: string, columnName: string, checked: boolean) => {
        if (checked) {
            setSelected((prev) => [...prev, { collectionName, columnName }]);
        } else {
            setSelected((prev) => prev.filter((item) => !(item.collectionName === collectionName && item.columnName === columnName)));
        }
    };

    const removeSelected = (collectionName: string, columnName: string) => {
        setSelected((prev) => prev.filter((item) => !(item.collectionName === collectionName && item.columnName === columnName)));
    };

    const handleSave = () => {
        if (!name.trim()) {
            toast.error("Nama konfigurasi tidak boleh kosong");
            return;
        }
        if (!startDate) {
            toast.error("Tanggal mulai peramalan wajib diisi");
            return;
        }
        mutate({ 
            name: name.trim(), 
            columns: selected,
            startDate: format(startDate, "yyyy-MM-dd"),
        });
    };

    const endDate = startDate ? new Date(startDate.getFullYear() + 1, startDate.getMonth(), startDate.getDate()) : null;

    const handleOpenChange = (isOpen: boolean) => {
        setOpen(isOpen);
        if (!isOpen) {
            setName("");
            setSelected([]);
            setStartDate(undefined);
        }
    };

    return (
        <TooltipProvider>
            <Dialog open={open} onOpenChange={handleOpenChange}>
                <DialogTrigger asChild>
                    <Button className="bg-gradient-to-r from-teal-600 to-emerald-600 hover:from-teal-700 hover:to-emerald-700 text-white text-base">
                        <Plus className="h-4 w-4 mr-2" />
                        Tambah Konfigurasi
                    </Button>
                </DialogTrigger>

                <DialogContent className="max-w-2xl w-[95vw] sm:w-full max-h-[85vh] flex flex-col p-0">
                    <DialogHeader className="px-4 sm:px-6 pt-4 sm:pt-6 pb-3 sm:pb-4">
                        <DialogTitle className="text-base sm:text-lg md:text-xl">Konfigurasi Peramalan Baru</DialogTitle>
                        <DialogDescription className="text-xs sm:text-sm md:text-base">
                            Pilih dataset dan kolom yang ingin diramalkan menggunakan model LSTM.
                        </DialogDescription>
                    </DialogHeader>

                    <div className="flex-1 overflow-hidden px-4 sm:px-6">
                        <div className="space-y-4 sm:space-y-5">
                            {/* Nama Konfigurasi */}
                            <div className="space-y-2">
                                <Label htmlFor="config-name" className="text-sm sm:text-base font-medium">
                                    Nama Konfigurasi
                                </Label>
                                <Input 
                                    id="config-name"
                                    placeholder="Contoh: Peramalan Cuaca Bulanan" 
                                    value={name} 
                                    onChange={(e) => setName(e.target.value)} 
                                    className="h-10"
                                />
                            </div>

                            {/* Tanggal Mulai Peramalan */}
                            <div className="space-y-2">
                                <div className="flex items-center gap-2">
                                    <Label className="text-base font-medium">Tanggal Mulai Peramalan</Label>
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
                                                "w-full justify-start text-left font-normal h-10 text-base", 
                                                !startDate && "text-muted-foreground"
                                            )}
                                        >
                                            <Calendar1 className="mr-2 h-4 w-4" />
                                            {startDate ? format(startDate, "PPP", { locale: id }) : "Pilih tanggal mulai"}
                                        </Button>
                                    </PopoverTrigger>
                                    <PopoverContent className="w-auto p-0" align="start">
                                        <Calendar 
                                            mode="single" 
                                            selected={startDate} 
                                            onSelect={setStartDate} 
                                            initialFocus 
                                            locale={id} 
                                        />
                                    </PopoverContent>
                                </Popover>

                                {/* Preview Tanggal Akhir */}
                                {endDate && (
                                    <p className="text-sm text-muted-foreground">
                                        Peramalan hingga: <span className="font-medium text-foreground">{format(endDate, "PPP", { locale: id })}</span>
                                    </p>
                                )}
                            </div>

                            <Separator />

                            {/* Dataset Selection */}
                            <div className="space-y-3">
                                <div className="flex items-center gap-2">
                                    <Label className="text-base font-medium">Pilih Kolom Dataset</Label>
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
                                        <p className="text-base text-muted-foreground">Tidak ada dataset tersedia</p>
                                        <p className="text-sm text-muted-foreground mt-1">Upload dataset terlebih dahulu</p>
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
                                                                <CardTitle className="text-base font-medium">
                                                                    {dataset.name || dataset.collectionName}
                                                                </CardTitle>
                                                            </div>
                                                            <Badge variant="outline" className="text-sm font-normal">
                                                                {dataset.columns?.length || 0} kolom
                                                            </Badge>
                                                        </div>
                                                    </CardHeader>
                                                    <CardContent className="pt-0 pb-3 px-4">
                                                        <div className="grid grid-cols-2 gap-1.5">
                                                            {dataset.columns?.map((col: string) => {
                                                                const isChecked = selected.some(
                                                                    (item) => item.collectionName === dataset.collectionName && item.columnName === col
                                                                );
                                                                const displayName = getDisplayName(col);

                                                                return (
                                                                    <Tooltip key={col}>
                                                                        <TooltipTrigger asChild>
                                                                            <label 
                                                                                className={`
                                                                                    flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer
                                                                                    text-base transition-all duration-150
                                                                                    ${isChecked 
                                                                                        ? 'bg-teal-50 text-teal-700 border border-teal-300 dark:bg-teal-950 dark:text-teal-300 dark:border-teal-800' 
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
                                                                            <p className="text-xs text-white">Kolom: <code className="bg-muted px-1 rounded text-foreground">{col}</code></p>
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
                            {selected.length > 0 && (
                                <div className="space-y-2 pb-2">
                                    <div className="flex items-center justify-between">
                                        <Label className="text-base font-medium">Kolom Terpilih</Label>
                                        <Badge variant="secondary" className="text-sm">
                                            {selected.length} dipilih
                                        </Badge>
                                    </div>
                                    <div className="flex flex-wrap gap-1.5 max-h-[80px] overflow-y-auto">
                                        {selected.map((item, index) => (
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
                        <Button 
                            variant="outline" 
                            onClick={() => handleOpenChange(false)}
                            disabled={isPending}
                            className="text-base"
                        >
                            Batal
                        </Button>
                        <Button 
                            onClick={handleSave} 
                            disabled={isPending || !name.trim() || selected.length === 0 || !startDate}
                            className="bg-gradient-to-r from-teal-600 to-emerald-600 hover:from-teal-700 hover:to-emerald-700 text-white text-base"
                        >
                            {isPending ? (
                                <>
                                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                    Menyimpan...
                                </>
                            ) : (
                                "Simpan Konfigurasi"
                            )}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </TooltipProvider>
    );
}