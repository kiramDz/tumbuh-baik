"use client"

import { useQuery } from "@tanstack/react-query"
import { CartesianGrid, Line, LineChart, XAxis, YAxis, Area, AreaChart, ComposedChart, Scatter, ScatterChart } from "recharts"
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { TrendingDown, TrendingUp, Minus, Calendar, Activity } from "lucide-react"
import { getLSTMDaily, getDecomposeLSTM } from "@/lib/fetch/files.fetch"
import { useState } from "react"
import axios from "axios"

const chartConfig = {
    value: {
        label: "Nilai",
        color: "hsl(var(--chart-1))",
    },
    historical: {
        label: "Historis",
        color: "hsl(var(--chart-2))",
    },
    trend: {
        label: "Trend",
        color: "hsl(var(--chart-3))",
    },
    seasonal: {
        label: "Seasonal",
        color: "hsl(var(--chart-4))",
    },
    residual: {
        label: "Residual",
        color: "hsl(var(--chart-5))",
    },
} satisfies ChartConfig

// Mapping nama parameter ke label yang lebih readable
const paramLabels: Record<string, string> = {
    "ALLSKY_SFC_SW_DWN": "Radiasi Matahari",
    "RH_AVG_preprocessed": "Kelembaban Udara",
    "RH_AVG": "Kelembaban Udara",
    "RH2M": "Kelembaban Udara",
    "TAVG": "Suhu Rata-rata",
    "TMAX": "Suhu Maksimum",
    "TMIN": "Suhu Minimum",
    "T2M": "Suhu Udara",
    "RR_imputed": "Curah Hujan",
    "PRECTOTCORR": "Curah Hujan",
    "RR": "Curah Hujan",
    "NDVI": "Indeks Vegetasi",
    "NDVI_imputed": "Indeks Vegetasi",
}

// Mapping unit untuk setiap parameter
const paramUnits: Record<string, string> = {
    "ALLSKY_SFC_SW_DWN": "MJ/m²",
    "RH_AVG_preprocessed": "%",
    "RH_AVG": "%",
    "RH2M": "%",
    "TAVG": "°C",
    "TMAX": "°C",
    "TMIN": "°C",
    "T2M": "°C",
    "RR_imputed": "mm",
    "RR": "mm",
    "PRECTOTCORR": "mm",
    "NDVI": "",
    "NDVI_imputed": "",
}

// Mapping parameter ke collection dan column name
const paramToCollection: Record<string, { collectionName: string; columnName: string }> = {
    // NASA Parameters - dari collection yang sama
    "ALLSKY_SFC_SW_DWN_AcehBesar": { 
        collectionName: "Nasa Aceh Besar Kec Darussalam", 
        columnName: "ALLSKY_SFC_SW_DWN" 
    },
    "ALLSKY_SFC_SW_DWN_AcehUtara": {
        collectionName: "Nasa Aceh Utara Kec Lhoksukon",
        columnName: "ALLSKY_SFC_SW_DWN"
    },
    "RH2M": { 
        collectionName: "Nasa Aceh Besar Kec Darussalam", 
        columnName: "RH2M" 
    },
    "RH2M_AcehUtara": {
        collectionName: "Nasa Aceh Utara Kec Lhoksukon",
        columnName: "RH2M"
    },
    "T2M": { 
        collectionName: "Nasa Aceh Besar Kec Darussalam", 
        columnName: "T2M" 
    },
    "T2M_AcehUtara": {
        collectionName: "Nasa Aceh Utara Kec Lhoksukon",
        columnName: "T2M"
    },
    "PRECTOTCORR": { 
        collectionName: "Nasa Aceh Besar Kec Darussalam", 
        columnName: "PRECTOTCORR" 
    },
    "PRECTOTCORR_AcehUtara": {
        collectionName: "Nasa Aceh Utara Kec Lhoksukon",
        columnName: "PRECTOTCORR"
    },
    "WS2M": { 
        collectionName: "Nasa Aceh Besar Kec Darussalam", 
        columnName: "WS2M" 
    },
    "PS": { 
        collectionName: "Nasa Aceh Besar Kec Darussalam", 
        columnName: "PS" 
    },
    
    // BMKG Parameters - dari collection berbeda
    "RH_AVG_preprocessed": { 
        collectionName: "kelembapan", 
        columnName: "RH_AVG_preprocessed" 
    },
    "TAVG": { 
        collectionName: "suhu bmkg", 
        columnName: "TAVG" 
    },
    "RR_imputed": { 
        collectionName: "rainfall", 
        columnName: "RR_imputed" 
    },
}

function getParamLabel(param: string): string {
    return paramLabels[param] || param
}

function getParamUnit(param: string): string {
    return paramUnits[param] || ""
}

function LoadingSkeleton() {
    return (
        <div className="space-y-4">
            <Skeleton className="h-10 w-64" />
            <div className="flex gap-2">
                {[...Array(4)].map((_, i) => (
                    <Skeleton key={i} className="h-9 w-24 rounded-lg" />
                ))}
            </div>
            <Card>
                <CardHeader>
                    <Skeleton className="h-6 w-48" />
                    <Skeleton className="h-4 w-32" />
                </CardHeader>
                <CardContent>
                    <Skeleton className="h-[300px] w-full rounded-lg" />
                </CardContent>
            </Card>
        </div>
    )
}

function EmptyState() {
    return (
        <Card className="border-dashed">
            <CardContent className="flex flex-col items-center justify-center py-16">
                <Activity className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="font-medium text-lg">Belum ada data peramalan</p>
                <p className="text-sm text-muted-foreground text-center max-w-sm mt-1">
                    Jalankan model peramalan terlebih dahulu untuk melihat grafik prediksi
                </p>
            </CardContent>
        </Card>
    )
}

interface ChartData {
    date: string
    fullDate: string
    value: number
    isHistorical?: boolean
    year: number
}

interface DecomposeData {
    date: string
    fullDate: string
    trend: number
    seasonal: number
    resid: number
    year: number
}

interface ParamChartProps {
    param: string
    data: ChartData[]
    decomposeData: DecomposeData[]
    mode: "forecast-only" | "combined" | "decomposed"
}

function ParamChart({ param, data, decomposeData, mode }: ParamChartProps) {
    if (mode === "decomposed") {
        // Render 3 Decomposed Charts Terpisah
        if (decomposeData.length === 0) {
            return (
                <Card>
                    <CardContent className="flex flex-col items-center justify-center py-16">
                        <Activity className="h-12 w-12 text-muted-foreground mb-4" />
                        <p className="font-medium text-lg">Tidak ada data decompose</p>
                        <p className="text-sm text-muted-foreground text-center max-w-sm mt-1">
                            Data decompose belum tersedia untuk parameter ini
                        </p>
                    </CardContent>
                </Card>
            )
        }

        const years = [...new Set(decomposeData.map(d => d.year))].sort()
        const yearRange = years.length
        const tickInterval = yearRange > 20 ? Math.ceil(decomposeData.length / 10) : Math.ceil(decomposeData.length / 15)
        
        const unit = getParamUnit(param)

        // Calculate stats untuk setiap komponen
        const trendValues = decomposeData.map(d => d.trend)
        const seasonalValues = decomposeData.map(d => d.seasonal)
        const residValues = decomposeData.map(d => d.resid)

        return (
            <div className="space-y-4">
                {/* Header Card */}
                <Card>
                    <CardHeader className="pb-4">
                        <div className="space-y-1">
                            <CardTitle className="text-xl font-semibold">
                                {getParamLabel(param)} - Decomposition Analysis
                            </CardTitle>
                            <CardDescription className="flex items-center gap-2">
                                <Calendar className="h-3.5 w-3.5" />
                                Data historis ({years[0]}-{years[years.length - 1]})
                            </CardDescription>
                        </div>
                        
                        {/* Stats Summary */}
                        <div className="grid grid-cols-3 gap-4 pt-4">
                            <div className="space-y-1">
                                <p className="text-xs font-medium text-[hsl(var(--chart-3))]">Trend</p>
                                <p className="text-sm font-medium tabular-nums">
                                    [{Math.min(...trendValues).toFixed(2)}, {Math.max(...trendValues).toFixed(2)}] {unit}
                                </p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-xs font-medium text-[hsl(var(--chart-4))]">Seasonal</p>
                                <p className="text-sm font-medium tabular-nums">
                                    [{Math.min(...seasonalValues).toFixed(2)}, {Math.max(...seasonalValues).toFixed(2)}] {unit}
                                </p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-xs font-medium text-[hsl(var(--chart-5))]">Residual</p>
                                <p className="text-sm font-medium tabular-nums">
                                    [{Math.min(...residValues).toFixed(2)}, {Math.max(...residValues).toFixed(2)}] {unit}
                                </p>
                            </div>
                        </div>
                    </CardHeader>
                </Card>

                {/* Chart 1: Trend - Line Chart */}
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-base font-semibold flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-[hsl(var(--chart-3))]"></div>
                            Trend Component
                        </CardTitle>
                        <CardDescription className="text-xs">
                            Long-term progression pattern
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ChartContainer config={chartConfig} className="h-[220px] w-full">
                            <LineChart 
                                data={decomposeData} 
                                margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                            >
                                <CartesianGrid 
                                    strokeDasharray="3 3" 
                                    vertical={false} 
                                    className="stroke-muted" 
                                />
                                <XAxis 
                                    dataKey="date" 
                                    tickLine={false} 
                                    axisLine={false} 
                                    tickMargin={8}
                                    interval={tickInterval}
                                    className="text-xs"
                                    tick={{ fill: 'hsl(var(--muted-foreground))' }}
                                />
                                <YAxis 
                                    tickLine={false} 
                                    axisLine={false}
                                    tickMargin={8}
                                    width={50}
                                    className="text-xs"
                                    tick={{ fill: 'hsl(var(--muted-foreground))' }}
                                    tickFormatter={(value) => value.toFixed(1)}
                                />
                                <ChartTooltip 
                                    cursor={{ stroke: 'hsl(var(--muted-foreground))', strokeWidth: 1 }}
                                    content={
                                        <ChartTooltipContent 
                                            formatter={(value) => [`${Number(value).toFixed(3)} ${unit}`, "Trend"]}
                                            labelFormatter={(label, payload) => {
                                                if (payload?.[0]?.payload?.fullDate) {
                                                    return payload[0].payload.fullDate
                                                }
                                                return label
                                            }}
                                        />
                                    } 
                                />
                                <Line
                                    type="monotone"
                                    dataKey="trend"
                                    stroke="hsl(var(--chart-3))"
                                    strokeWidth={2.5}
                                    dot={false}
                                />
                            </LineChart>
                        </ChartContainer>
                    </CardContent>
                </Card>

                {/* Chart 2: Seasonal - Area Chart */}
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-base font-semibold flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-[hsl(var(--chart-4))]"></div>
                            Seasonal Component
                        </CardTitle>
                        <CardDescription className="text-xs">
                            Repeating patterns and cycles
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ChartContainer config={chartConfig} className="h-[220px] w-full">
                            <AreaChart 
                                data={decomposeData} 
                                margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                            >
                                <defs>
                                    <linearGradient id={`gradient-seasonal-${param}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="hsl(var(--chart-4))" stopOpacity={0.4} />
                                        <stop offset="100%" stopColor="hsl(var(--chart-4))" stopOpacity={0.05} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid 
                                    strokeDasharray="3 3" 
                                    vertical={false} 
                                    className="stroke-muted" 
                                />
                                <XAxis 
                                    dataKey="date" 
                                    tickLine={false} 
                                    axisLine={false} 
                                    tickMargin={8}
                                    interval={tickInterval}
                                    className="text-xs"
                                    tick={{ fill: 'hsl(var(--muted-foreground))' }}
                                />
                                <YAxis 
                                    tickLine={false} 
                                    axisLine={false}
                                    tickMargin={8}
                                    width={50}
                                    className="text-xs"
                                    tick={{ fill: 'hsl(var(--muted-foreground))' }}
                                    tickFormatter={(value) => value.toFixed(1)}
                                />
                                <ChartTooltip 
                                    cursor={{ stroke: 'hsl(var(--muted-foreground))', strokeWidth: 1 }}
                                    content={
                                        <ChartTooltipContent 
                                            formatter={(value) => [`${Number(value).toFixed(3)} ${unit}`, "Seasonal"]}
                                            labelFormatter={(label, payload) => {
                                                if (payload?.[0]?.payload?.fullDate) {
                                                    return payload[0].payload.fullDate
                                                }
                                                return label
                                            }}
                                        />
                                    } 
                                />
                                <Area
                                    type="monotone"
                                    dataKey="seasonal"
                                    stroke="hsl(var(--chart-4))"
                                    strokeWidth={2}
                                    fill={`url(#gradient-seasonal-${param})`}
                                    dot={false}
                                />
                            </AreaChart>
                        </ChartContainer>
                    </CardContent>
                </Card>

                {/* Chart 3: Residual - Scatter Chart */}
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-base font-semibold flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-[hsl(var(--chart-5))]"></div>
                            Residual Component
                        </CardTitle>
                        <CardDescription className="text-xs">
                            Random variations and noise
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ChartContainer config={chartConfig} className="h-[220px] w-full">
                            <ScatterChart 
                                data={decomposeData} 
                                margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                            >
                                <CartesianGrid 
                                    strokeDasharray="3 3" 
                                    vertical={false} 
                                    className="stroke-muted" 
                                />
                                <XAxis 
                                    dataKey="date" 
                                    tickLine={false} 
                                    axisLine={false} 
                                    tickMargin={8}
                                    interval={tickInterval}
                                    className="text-xs"
                                    tick={{ fill: 'hsl(var(--muted-foreground))' }}
                                />
                                <YAxis 
                                    tickLine={false} 
                                    axisLine={false}
                                    tickMargin={8}
                                    width={50}
                                    className="text-xs"
                                    tick={{ fill: 'hsl(var(--muted-foreground))' }}
                                    tickFormatter={(value) => value.toFixed(1)}
                                />
                                <ChartTooltip 
                                    cursor={{ stroke: 'hsl(var(--muted-foreground))', strokeWidth: 1 }}
                                    content={
                                        <ChartTooltipContent 
                                            formatter={(value) => [`${Number(value).toFixed(3)} ${unit}`, "Residual"]}
                                            labelFormatter={(label, payload) => {
                                                if (payload?.[0]?.payload?.fullDate) {
                                                    return payload[0].payload.fullDate
                                                }
                                                return label
                                            }}
                                        />
                                    } 
                                />
                                {/* Zero line reference */}
                                <Line 
                                    type="monotone" 
                                    dataKey={() => 0} 
                                    stroke="hsl(var(--muted-foreground))" 
                                    strokeWidth={1}
                                    strokeDasharray="5 5"
                                    dot={false}
                                />
                                <Scatter
                                    dataKey="resid"
                                    fill="hsl(var(--chart-5))"
                                    fillOpacity={0.7}
                                />
                            </ScatterChart>
                        </ChartContainer>
                    </CardContent>
                </Card>
            </div>
        )
    }

    // Original Forecast/Combined Chart
    if (data.length === 0) return null

    const forecastData = data.filter(d => !d.isHistorical)
    const historicalData = data.filter(d => d.isHistorical)

    const displayData = mode === "forecast-only" ? forecastData : data

    const firstValue = displayData[0].value
    const lastValue = displayData[displayData.length - 1].value
    const percentChange = firstValue !== 0 ? ((lastValue - firstValue) / firstValue) * 100 : 0
    
    const minValue = Math.min(...displayData.map(d => d.value))
    const maxValue = Math.max(...displayData.map(d => d.value))
    const avgValue = displayData.reduce((sum, d) => sum + d.value, 0) / displayData.length

    const isUp = percentChange > 1
    const isDown = percentChange < -1
    const TrendIcon = isDown ? TrendingDown : isUp ? TrendingUp : Minus
    
    const trendVariant = isDown ? "destructive" : isUp ? "default" : "secondary"
    const unit = getParamUnit(param)

    const years = [...new Set(displayData.map(d => d.year))].sort()
    const yearRange = years.length
    
    let tickInterval: number
    if (mode === "combined") {
        if (yearRange > 20) {
            tickInterval = Math.ceil(displayData.length / 10)
        } else if (yearRange > 10) {
            tickInterval = Math.ceil(displayData.length / 15)
        } else {
            tickInterval = Math.ceil(displayData.length / 20)
        }
    } else {
        tickInterval = Math.ceil(displayData.length / 12)
    }

    return (
        <Card>
            <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                    <div className="space-y-1">
                        <CardTitle className="text-xl font-semibold">
                            {getParamLabel(param)}
                        </CardTitle>
                        <CardDescription className="flex items-center gap-2">
                            <Calendar className="h-3.5 w-3.5" />
                            {mode === "combined" 
                                ? `Data historis (${years[0]}-${years[years.length - 1]}) & prediksi` 
                                : `Prediksi 365 hari ke depan`
                            }
                        </CardDescription>
                    </div>
                    <Badge variant={trendVariant} className="flex items-center gap-1">
                        <TrendIcon className="h-3.5 w-3.5" />
                        {percentChange >= 0 ? "+" : ""}{percentChange.toFixed(1)}%
                    </Badge>
                </div>
                
                <div className="grid grid-cols-3 gap-4 pt-4">
                    <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Minimum</p>
                        <p className="text-sm font-medium tabular-nums">
                            {minValue.toFixed(2)} {unit}
                        </p>
                    </div>
                    <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Rata-rata</p>
                        <p className="text-sm font-medium tabular-nums">
                            {avgValue.toFixed(2)} {unit}
                        </p>
                    </div>
                    <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Maksimum</p>
                        <p className="text-sm font-medium tabular-nums">
                            {maxValue.toFixed(2)} {unit}
                        </p>
                    </div>
                </div>
            </CardHeader>

            <CardContent className="pt-4">
                <ChartContainer config={chartConfig} className="h-[280px] w-full">
                    <AreaChart 
                        data={displayData} 
                        margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                    >
                        <defs>
                            <linearGradient id={`gradient-forecast-${param}`} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="hsl(var(--chart-1))" stopOpacity={0.3} />
                                <stop offset="100%" stopColor="hsl(var(--chart-1))" stopOpacity={0.05} />
                            </linearGradient>
                            <linearGradient id={`gradient-historical-${param}`} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="hsl(var(--chart-2))" stopOpacity={0.3} />
                                <stop offset="100%" stopColor="hsl(var(--chart-2))" stopOpacity={0.05} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid 
                            strokeDasharray="3 3" 
                            vertical={false} 
                            className="stroke-muted" 
                        />
                        <XAxis 
                            dataKey="date" 
                            tickLine={false} 
                            axisLine={false} 
                            tickMargin={8}
                            interval={tickInterval}
                            className="text-xs"
                            tick={{ fill: 'hsl(var(--muted-foreground))' }}
                        />
                        <YAxis 
                            tickLine={false} 
                            axisLine={false}
                            tickMargin={8}
                            width={50}
                            className="text-xs"
                            tick={{ fill: 'hsl(var(--muted-foreground))' }}
                            tickFormatter={(value) => value.toFixed(1)}
                        />
                        <ChartTooltip 
                            cursor={{ stroke: 'hsl(var(--muted-foreground))', strokeWidth: 1 }}
                            content={
                                <ChartTooltipContent 
                                    formatter={(value, name, item) => {
                                        const label = item.payload.isHistorical ? "Historis" : "Prediksi"
                                        return [`${Number(value).toFixed(2)} ${unit}`, label]
                                    }}
                                    labelFormatter={(label, payload) => {
                                        if (payload?.[0]?.payload?.fullDate) {
                                            return payload[0].payload.fullDate
                                        }
                                        return label
                                    }}
                                />
                            } 
                        />
                        {mode === "combined" && historicalData.length > 0 && (
                            <Area
                                type="monotone"
                                dataKey={(item) => item.isHistorical ? item.value : null}
                                stroke="hsl(var(--chart-2))"
                                strokeWidth={2}
                                fill={`url(#gradient-historical-${param})`}
                                dot={false}
                                activeDot={{ r: 4, fill: 'hsl(var(--chart-2))' }}
                                connectNulls={false}
                            />
                        )}
                        <Area
                            type="monotone"
                            dataKey={(item) => mode === "combined" ? (!item.isHistorical ? item.value : null) : item.value}
                            stroke="hsl(var(--chart-1))"
                            strokeWidth={2}
                            fill={`url(#gradient-forecast-${param})`}
                            dot={false}
                            activeDot={{ r: 4, fill: 'hsl(var(--chart-1))' }}
                            connectNulls={mode === "forecast-only"}
                        />
                    </AreaChart>
                </ChartContainer>
            </CardContent>
        </Card>
    )
}

// Fungsi untuk fetch SEMUA data historis dari collection
async function fetchHistoricalData(collectionName: string, columnName: string) {
    try {
        const countResponse = await axios.get(`/api/v1/dataset-meta/${collectionName}`, {
            params: { 
                page: 1, 
                pageSize: 10,
                sortBy: "Date",
                sortOrder: "asc"
            }
        })
        
        const total = countResponse.data?.data?.total || 0
        if (total === 0) return []

        const response = await axios.get(`/api/v1/dataset-meta/${collectionName}`, {
            params: { 
                page: 1, 
                pageSize: total,
                sortBy: "Date",
                sortOrder: "asc"
            }
        })
        
        const items = response.data?.data?.items || []
        return items
            .filter((item: any) => item.Date && item[columnName] != null)
            .map((item: any) => ({
                date: item.Date,
                value: item[columnName]
            }))
    } catch (error) {
        console.error(`Error fetching historical data for ${collectionName}:`, error)
        return []
    }
}

export function LSTMLineChart() {
    const [viewMode, setViewMode] = useState<"forecast-only" | "combined" | "decomposed">("forecast-only")

    const { data: rawData, isLoading } = useQuery({
        queryKey: ["lstm-daily-full"],
        queryFn: () => getLSTMDaily(1, 365),
        refetchOnWindowFocus: false,
    })

    const { data: historicalDataMap, isLoading: isLoadingHistorical } = useQuery({
        queryKey: ["lstm-historical-data", viewMode],
        queryFn: async () => {
            if (viewMode !== "combined") return {}

            const items = rawData?.items || []
            if (items.length === 0) return {}

            const parameters = new Set<string>()
            items.forEach((item: any) => {
                Object.keys(item.parameters || {}).forEach((param) => parameters.add(param))
            })

            const paramArray = Array.from(parameters)
            const historicalMap: Record<string, any[]> = {}

            await Promise.all(
                paramArray.map(async (param) => {
                    const mapping = paramToCollection[param]
                    if (mapping) {
                        const data = await fetchHistoricalData(
                            mapping.collectionName, 
                            mapping.columnName
                        )
                        historicalMap[param] = data
                    }
                })
            )

            return historicalMap
        },
        enabled: viewMode === "combined" && !!rawData,
        refetchOnWindowFocus: false,
    })

    const { data: decomposeRawData, isLoading: isLoadingDecompose } = useQuery({
        queryKey: ["lstm-decompose-data", viewMode],
        queryFn: getDecomposeLSTM,
        enabled: viewMode === "decomposed",
        refetchOnWindowFocus: false,
    })

    const loading = isLoading || 
                    (viewMode === "combined" && isLoadingHistorical) ||
                    (viewMode === "decomposed" && isLoadingDecompose)

    if (loading) return <LoadingSkeleton />

    const items = rawData?.items || []
    if (items.length === 0 && viewMode !== "decomposed") return <EmptyState />

    const parameters = new Set<string>()
    if (viewMode === "decomposed") {
        decomposeRawData?.forEach((item: any) => {
            Object.keys(item.parameters || {}).forEach((param) => parameters.add(param))
        })
    } else {
        items.forEach((item: any) => {
            Object.keys(item.parameters || {}).forEach((param) => parameters.add(param))
        })
    }

    const paramArray = Array.from(parameters)

    const groupedData: Record<string, ChartData[]> = {}
    paramArray.forEach((param) => {
        const forecastData = items
            .filter((item: any) => item.parameters?.[param]?.forecast_value != null)
            .map((item: any) => {
                const dateObj = new Date(item.forecast_date)
                return {
                    date: dateObj.getFullYear().toString(),
                    fullDate: dateObj.toLocaleDateString("id-ID", { 
                        weekday: "long", 
                        year: "numeric", 
                        month: "long", 
                        day: "numeric" 
                    }),
                    value: item.parameters[param].forecast_value,
                    isHistorical: false,
                    year: dateObj.getFullYear(),
                }
            })

        if (viewMode === "combined" && historicalDataMap?.[param]) {
            const historical = historicalDataMap[param].map((item: any) => {
                const dateObj = new Date(item.date)
                return {
                    date: dateObj.getFullYear().toString(),
                    fullDate: dateObj.toLocaleDateString("id-ID", { 
                        weekday: "long", 
                        year: "numeric", 
                        month: "long", 
                        day: "numeric" 
                    }),
                    value: item.value,
                    isHistorical: true,
                    year: dateObj.getFullYear(),
                }
            })
            groupedData[param] = [...historical, ...forecastData]
        } else {
            groupedData[param] = forecastData
        }

        groupedData[param].sort((a: any, b: any) => {
            return new Date(a.fullDate).getTime() - new Date(b.fullDate).getTime()
        })
    })

    const groupedDecomposeData: Record<string, DecomposeData[]> = {}
    if (viewMode === "decomposed" && decomposeRawData) {
        paramArray.forEach((param) => {
            const decomposeData = decomposeRawData
                .filter((item: any) => item.parameters?.[param])
                .map((item: any) => {
                    const dateObj = new Date(item.date)
                    return {
                        date: dateObj.getFullYear().toString(),
                        fullDate: dateObj.toLocaleDateString("id-ID", { 
                            weekday: "long", 
                            year: "numeric", 
                            month: "long", 
                            day: "numeric" 
                        }),
                        trend: item.parameters[param].trend,
                        seasonal: item.parameters[param].seasonal,
                        resid: item.parameters[param].resid,
                        year: dateObj.getFullYear(),
                    }
                })
                .sort((a: any, b: any) => new Date(a.fullDate).getTime() - new Date(b.fullDate).getTime())
            
            groupedDecomposeData[param] = decomposeData
        })
    }

    if (paramArray.length === 0) return <EmptyState />

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-4">
                <div className="space-y-2 flex-1 max-w-xs">
                    <Label htmlFor="view-mode">Mode Tampilan</Label>
                    <Select value={viewMode} onValueChange={(value: any) => setViewMode(value)}>
                        <SelectTrigger id="view-mode">
                            <SelectValue placeholder="Pilih mode tampilan" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="forecast-only">Hanya Peramalan</SelectItem>
                            <SelectItem value="combined">Historis + Peramalan</SelectItem>
                            <SelectItem value="decomposed">Decomposed</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
            </div>

            <Tabs defaultValue={paramArray[0]} className="w-full">
                <TabsList className="mb-4 flex-wrap h-auto gap-2 bg-transparent p-0">
                    {paramArray.map((param) => (
                        <TabsTrigger 
                            key={param} 
                            value={param}
                            className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-lg px-4 py-2 border"
                        >
                            {getParamLabel(param)}
                        </TabsTrigger>
                    ))}
                </TabsList>

                {paramArray.map((param) => (
                    <TabsContent key={param} value={param} className="mt-0">
                        <ParamChart 
                            param={param} 
                            data={groupedData[param] || []} 
                            decomposeData={groupedDecomposeData[param] || []}
                            mode={viewMode} 
                        />
                    </TabsContent>
                ))}
            </Tabs>
        </div>
    )
}
