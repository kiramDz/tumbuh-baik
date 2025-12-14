"use client"

import { useQuery } from "@tanstack/react-query"
import { getLSTMConfigs } from "@/lib/fetch/files.fetch"
import { LabelList, Pie, PieChart } from "recharts"
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { PieChartIcon } from "lucide-react"

const COLORS = {
    mae: "hsl(var(--chart-1))",
    rmse: "hsl(var(--chart-2))",
    mape: "hsl(var(--chart-5))", // Ubah dari chart-3 ke chart-5 (lebih terang)
}

const chartConfig = {
    mae: { label: "MAE", color: COLORS.mae },
    rmse: { label: "RMSE", color: COLORS.rmse },
    mape: { label: "MAPE", color: COLORS.mape },
} satisfies ChartConfig

// Fungsi helper yang menerima collectionName dan columnName
const getDisplayName = (columnName: string, collectionName?: string): string => {
    // Debug: Log input
    console.log('🔍 getDisplayName called:', { columnName, collectionName });
    
    // Cek kombinasi collection + column dulu
    const combinedKey = collectionName ? `${collectionName}::${columnName}` : null;
    console.log('🔑 Combined key:', combinedKey);
    
    if (combinedKey && COLUMN_NAME_MAPPING[combinedKey]) {
        console.log('✅ Found in combined mapping:', COLUMN_NAME_MAPPING[combinedKey]);
        return COLUMN_NAME_MAPPING[combinedKey];
    }
    
    // Fallback ke columnName saja
    if (COLUMN_NAME_MAPPING[columnName]) {
        console.log('✅ Found in column mapping:', COLUMN_NAME_MAPPING[columnName]);
        return COLUMN_NAME_MAPPING[columnName];
    }
    
    // Fallback formatting default
    const formatted = columnName
        .replace(/_/g, ' ')
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
    
    console.log('⚠️ Using default formatting:', formatted);
    return formatted;
};

// Mapping nama kolom dengan format: "collectionName::columnName"
const COLUMN_NAME_MAPPING: Record<string, string> = {
    // ============================================
    // BMKG - Kelembapan
    // ============================================
    'kelembapan::RH_AVG_preprocessed': 'Kelembaban Rata-rata (BMKG)',
    'RH_AVG_preprocessed': 'Kelembaban Rata-rata',
    'RH_AVG': 'Kelembaban Rata-rata',

    // ============================================
    // BMKG - Suhu
    // ============================================
    'suhu bmkg::TAVG': 'Suhu Rata-rata (BMKG)',
    'suhu bmkg::TMAX': 'Suhu Maksimum (BMKG)',
    'suhu bmkg::TMIN': 'Suhu Minimum (BMKG)',
    'TN': 'Suhu Minimum',
    'TX': 'Suhu Maksimum',
    'TAVG': 'Suhu Rata-rata',
    'TMAX': 'Suhu Maksimum',
    'TMIN': 'Suhu Minimum',

    // ============================================
    // BMKG - Curah Hujan
    // ============================================
    'rainfall::RR_imputed': 'Curah Hujan (BMKG)',
    'RR': 'Curah Hujan',
    'RR_original': 'Curah Hujan Asli',
    'RR_imputed': 'Curah Hujan (Diperbaiki)',
    'is_outlier': 'Status Outlier',
    'RR_log': 'Curah Hujan (Log)',
    'RR_sqrt': 'Curah Hujan (Akar)',
    'RR_boxcox': 'Curah Hujan (Box-Cox)',

    // ============================================
    // NASA Aceh Besar Kec Darussalam
    // ============================================
    'Nasa Aceh Besar Kec Darussalam::ALLSKY_SFC_SW_DWN': 'Radiasi Matahari (Aceh Besar)',
    'Nasa Aceh Besar Kec Darussalam::RH2M': 'Kelembaban Udara (Aceh Besar)',
    'Nasa Aceh Besar Kec Darussalam::T2M': 'Suhu Udara (Aceh Besar)',
    'Nasa Aceh Besar Kec Darussalam::T2M_MAX': 'Suhu Maksimum (Aceh Besar)',
    'Nasa Aceh Besar Kec Darussalam::T2M_MIN': 'Suhu Minimum (Aceh Besar)',
    'Nasa Aceh Besar Kec Darussalam::PRECTOTCORR': 'Curah Hujan (Aceh Besar)',
    'Nasa Aceh Besar Kec Darussalam::WS2M': 'Kecepatan Angin 2m (Aceh Besar)',
    'Nasa Aceh Besar Kec Darussalam::WS10M': 'Kecepatan Angin 10m (Aceh Besar)',
    'Nasa Aceh Besar Kec Darussalam::WS10M_MAX': 'Angin Maksimum (Aceh Besar)',
    'Nasa Aceh Besar Kec Darussalam::PS': 'Tekanan Permukaan (Aceh Besar)',
    'Nasa Aceh Besar Kec Darussalam::QV2M': 'Kelembaban Spesifik (Aceh Besar)',

    // ============================================
    // NASA Aceh Utara Kec Lhoksukon (PERHATIKAN: NASA huruf kapital semua!)
    // ============================================
    'NASA Aceh Utara Kec Lhoksukon::ALLSKY_SFC_SW_DWN': 'Radiasi Matahari (Aceh Utara)',
    'NASA Aceh Utara Kec Lhoksukon::RH2M': 'Kelembaban Udara (Aceh Utara)',
    'NASA Aceh Utara Kec Lhoksukon::T2M': 'Suhu Udara (Aceh Utara)',
    'NASA Aceh Utara Kec Lhoksukon::T2M_MAX': 'Suhu Maksimum (Aceh Utara)',
    'NASA Aceh Utara Kec Lhoksukon::T2M_MIN': 'Suhu Minimum (Aceh Utara)',
    'NASA Aceh Utara Kec Lhoksukon::PRECTOTCORR': 'Curah Hujan (Aceh Utara)',
    'NASA Aceh Utara Kec Lhoksukon::WS2M': 'Kecepatan Angin 2m (Aceh Utara)',
    'NASA Aceh Utara Kec Lhoksukon::WS10M': 'Kecepatan Angin 10m (Aceh Utara)',
    'NASA Aceh Utara Kec Lhoksukon::PS': 'Tekanan Permukaan (Aceh Utara)',
    'NASA Aceh Utara Kec Lhoksukon::QV2M': 'Kelembaban Spesifik (Aceh Utara)',

    // ============================================
    // NASA Generic (fallback tanpa lokasi)
    // ============================================
    'T2M': 'Suhu Udara',
    'T2M_MAX': 'Suhu Maksimum',
    'T2M_MIN': 'Suhu Minimum',
    'RH2M': 'Kelembaban Udara',
    'PRECTOTCORR': 'Curah Hujan',
    'ALLSKY_SFC_SW_DWN': 'Radiasi Matahari',
    'WS2M': 'Kecepatan Angin 2m',
    'WS10M': 'Kecepatan Angin 10m',
    'WS10M_MAX': 'Angin Maksimum',
    'WD10M': 'Arah Angin',
    'PS': 'Tekanan Permukaan',
    'QV2M': 'Kelembaban Spesifik',

    // ============================================
    // NDVI
    // ============================================
    'NDVI': 'Indeks Vegetasi',
    'NDVI_imputed': 'Indeks Vegetasi (Diperbaiki)',

    // ============================================
    // Umum
    // ============================================
    'Date': 'Tanggal',
    'Year': 'Tahun',
    'Month': 'Bulan',
    'Day': 'Hari',
    'month': 'Bulan',
    'day': 'Hari',
};

export function LSTMPieChart() {
    const { data = [], isLoading } = useQuery({
        queryKey: ["lstm-config"],
        queryFn: getLSTMConfigs,
    })

    // Debug: Log raw data
    console.log('📊 LSTM Config Data:', data);

    if (isLoading) {
        return (
            <div className="grid md:grid-cols-2 gap-4">
                {[...Array(2)].map((_, i) => (
                    <Card key={i}>
                        <CardHeader className="pb-2">
                            <Skeleton className="h-5 w-3/4" />
                        </CardHeader>
                        <CardContent>
                            <Skeleton className="h-[200px] w-full rounded-lg" />
                        </CardContent>
                    </Card>
                ))}
            </div>
        )
    }

    const completed = data.filter((item: any) => item.status === "done" && item.error_metrics?.length > 0)
    console.log('✅ Completed forecasts:', completed);

    if (completed.length === 0) {
        return (
            <Card className="border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-10">
                    <PieChartIcon className="h-10 w-10 text-muted-foreground mb-3" />
                    <p className="font-medium">Belum ada data metrik</p>
                    <p className="text-sm text-muted-foreground">
                        Jalankan peramalan untuk melihat metrik error
                    </p>
                </CardContent>
            </Card>
        )
    }

    const errorMetricsArray = completed[0]?.error_metrics ?? []
    console.log('📈 Error Metrics Array:', errorMetricsArray);

    return (
        <TooltipProvider>
            <div className="grid md:grid-cols-2 gap-4">
                {errorMetricsArray.map((entry: any, index: number) => {
                    console.log(`📌 Processing entry ${index}:`, entry);
                    
                    const metrics = entry.metrics_lstm ?? {}  // Ubah dari entry.metrics ke entry.metrics_lstm
                    const chartData = [
                        { key: "mae", value: metrics.mae || 0, fill: COLORS.mae },
                        { key: "rmse", value: metrics.rmse || 0, fill: COLORS.rmse },
                        { key: "mape", value: metrics.mape || 0, fill: COLORS.mape },
                    ]

                    const displayName = getDisplayName(entry.columnName, entry.collectionName)
                    console.log('🏷️ Final display name:', displayName);

                    return (
                        <Card key={index}>
                            <CardHeader className="pb-2">
                                <div className="flex items-center justify-between">
                                    <Tooltip>
                                        <TooltipTrigger asChild>
                                            <CardTitle className="text-base font-medium cursor-help">
                                                {displayName}
                                            </CardTitle>
                                        </TooltipTrigger>
                                        <TooltipContent side="top">
                                            <p className="text-xs">
                                                Kolom: <code className="bg-muted px-1 rounded">{entry.columnName}</code>
                                            </p>
                                            <p className="text-xs">
                                                Collection: <code className="bg-muted px-1 rounded">{entry.collectionName}</code>
                                            </p>
                                        </TooltipContent>
                                    </Tooltip>
                                    <Badge variant="outline" className="text-xs font-normal">
                                        AIC: {(metrics.aic || 0).toExponential(2)}
                                    </Badge>
                                </div>
                                <p className="text-xs text-muted-foreground">
                                    {entry.collectionName}
                                </p>
                            </CardHeader>

                            <CardContent className="pt-0">
                                <div className="flex items-center gap-4">
                                    <ChartContainer config={chartConfig} className="mx-auto aspect-square h-[180px]">
                                        <PieChart>
                                            <ChartTooltip content={<ChartTooltipContent nameKey="key" hideLabel />} />
                                            <Pie 
                                                data={chartData} 
                                                dataKey="value" 
                                                nameKey="key"
                                                innerRadius={40} 
                                                outerRadius={70}
                                                cornerRadius={4} 
                                                paddingAngle={2}
                                            >
                                                <LabelList 
                                                    dataKey="value" 
                                                    stroke="none" 
                                                    fontSize={10} 
                                                    fontWeight={500}
                                                    formatter={(value: number) => value.toFixed(1)} 
                                                />
                                            </Pie>
                                        </PieChart>
                                    </ChartContainer>

                                    <div className="flex flex-col gap-2">
                                        {chartData.map((item) => (
                                            <div key={item.key} className="flex items-center gap-2">
                                                <div 
                                                    className="h-3 w-3 rounded-sm shrink-0" 
                                                    style={{ backgroundColor: item.fill }} 
                                                />
                                                <div className="flex items-center gap-2">
                                                    <span className="text-xs text-muted-foreground uppercase w-10">
                                                        {item.key}
                                                    </span>
                                                    <span className="text-xs font-medium tabular-nums">
                                                        {item.value.toFixed(2)}
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    )
                })}
            </div>
        </TooltipProvider>
    )
}
