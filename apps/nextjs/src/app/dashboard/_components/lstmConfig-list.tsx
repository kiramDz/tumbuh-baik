"use client"

import { Card, CardContent } from "@/components/ui/card"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useQuery } from "@tanstack/react-query"
import { getLSTMConfigs } from "@/lib/fetch/files.fetch"
import { CheckCircle2, AlertTriangle, Clock, Loader2, Database } from "lucide-react"
import { formatDistanceToNow } from "date-fns"
import { id } from "date-fns/locale"

interface LSTMConfig {
    _id: string;
    name: string;
    status: "done" | "failed" | "pending" | "running";
    columns?: { collectionName: string; columnName: string }[];
    errorMessage?: string;
    createdAt?: string;
}

// Mapping nama kolom
const COLUMN_NAME_MAPPING: Record<string, string> = {
    // Kelembapan
    'RH_AVG_preprocessed': 'Kelembaban Rata-rata',

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

export function LSTMConfigList() {
    const { data = [], isLoading } = useQuery<LSTMConfig[]>({
        queryKey: ["lstm-config"],
        queryFn: getLSTMConfigs,
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
                    <p className="font-medium">Belum ada konfigurasi</p>
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

function ConfigCard({ config }: { config: LSTMConfig }) {
    const status = getStatus(config.status);
    const StatusIcon = status.icon;

    return (
        <Card className="w-[280px] shrink-0 hover:shadow-md transition-shadow">
            <CardContent className="p-4">
                {/* Header */}
                <div className="mb-3">
                    <h4 className="font-medium text-sm truncate mb-1">{config.name}</h4>
                    <div className="flex items-center gap-1.5">
                        <StatusIcon className={`h-3.5 w-3.5 ${status.color}`} />
                        <span className={`text-xs ${status.color}`}>{status.label}</span>
                        {config.createdAt && (
                            <>
                                <span className="text-muted-foreground">·</span>
                                <span className="text-xs text-muted-foreground">
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
                                <Badge variant="secondary" className="text-xs font-normal cursor-help">
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
                                <Badge variant="outline" className="text-xs font-normal cursor-help">
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
                    <div className="flex items-center gap-2 text-blue-600 text-xs">
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        <span>Memproses...</span>
                    </div>
                )}

                {config.errorMessage && (
                    <p className="text-xs text-destructive line-clamp-1">
                        {config.errorMessage}
                    </p>
                )}
            </CardContent>
        </Card>
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