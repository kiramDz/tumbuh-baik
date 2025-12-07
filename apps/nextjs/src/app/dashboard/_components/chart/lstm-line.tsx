"use client"

import { useQuery } from "@tanstack/react-query"
import { CartesianGrid, Line, LineChart, XAxis, YAxis, Area, AreaChart } from "recharts"
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TrendingDown, TrendingUp, Minus, Calendar, Activity } from "lucide-react"
import { getLSTMDaily } from "@/lib/fetch/files.fetch"

const chartConfig = {
    value: {
        label: "Nilai",
        color: "hsl(var(--chart-1))",
    },
} satisfies ChartConfig

// Mapping nama parameter ke label yang lebih readable
const paramLabels: Record<string, string> = {
    "ALLSKY_SFC_SW_DWN": "Radiasi Matahari",
    "RH_AVG_preprocessed": "Kelembaban Udara",
    "RH_AVG": "Kelembaban Udara",
    "TAVG": "Suhu Rata-rata",
    "TMAX": "Suhu Maksimum",
    "TMIN": "Suhu Minimum",
    "RR_imputed": "Curah Hujan",
    "RR": "Curah Hujan",
    "NDVI": "Indeks Vegetasi",
    "NDVI_imputed": "Indeks Vegetasi",
}

// Mapping unit untuk setiap parameter
const paramUnits: Record<string, string> = {
    "ALLSKY_SFC_SW_DWN": "MJ/m²",
    "RH_AVG_preprocessed": "%",
    "RH_AVG": "%",
    "TAVG": "°C",
    "TMAX": "°C",
    "TMIN": "°C",
    "RR_imputed": "mm",
    "RR": "mm",
    "NDVI": "",
    "NDVI_imputed": "",
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
}

interface ParamChartProps {
    param: string
    data: ChartData[]
}

function ParamChart({ param, data }: ParamChartProps) {
    if (data.length === 0) return null

    const firstValue = data[0].value
    const lastValue = data[data.length - 1].value
    const percentChange = firstValue !== 0 ? ((lastValue - firstValue) / firstValue) * 100 : 0
    
    const minValue = Math.min(...data.map(d => d.value))
    const maxValue = Math.max(...data.map(d => d.value))
    const avgValue = data.reduce((sum, d) => sum + d.value, 0) / data.length

    const isUp = percentChange > 1
    const isDown = percentChange < -1
    const TrendIcon = isDown ? TrendingDown : isUp ? TrendingUp : Minus
    
    const trendVariant = isDown ? "destructive" : isUp ? "default" : "secondary"
    const unit = getParamUnit(param)

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
                            Prediksi 365 hari ke depan
                        </CardDescription>
                    </div>
                    <Badge variant={trendVariant} className="flex items-center gap-1">
                        <TrendIcon className="h-3.5 w-3.5" />
                        {percentChange >= 0 ? "+" : ""}{percentChange.toFixed(1)}%
                    </Badge>
                </div>
                
                {/* Stats Summary */}
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
                        data={data} 
                        margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                    >
                        <defs>
                            <linearGradient id={`gradient-${param}`} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="hsl(var(--chart-1))" stopOpacity={0.3} />
                                <stop offset="100%" stopColor="hsl(var(--chart-1))" stopOpacity={0.05} />
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
                            interval="preserveStartEnd"
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
                                    formatter={(value) => [`${Number(value).toFixed(2)} ${unit}`, getParamLabel(param)]}
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
                            dataKey="value"
                            stroke="hsl(var(--chart-1))"
                            strokeWidth={2}
                            fill={`url(#gradient-${param})`}
                            dot={false}
                            activeDot={{ r: 4, fill: 'hsl(var(--chart-1))' }}
                        />
                    </AreaChart>
                </ChartContainer>
            </CardContent>
        </Card>
    )
}

export function LSTMLineChart() {
    const { data: rawData, isLoading } = useQuery({
        queryKey: ["lstm-daily-full"],
        queryFn: () => getLSTMDaily(1, 365),
        refetchOnWindowFocus: false,
    })

    if (isLoading) return <LoadingSkeleton />

    const items = rawData?.items || []
    if (items.length === 0) return <EmptyState />

    const parameters = new Set<string>()
    items.forEach((item: any) => {
        Object.keys(item.parameters || {}).forEach((param) => parameters.add(param))
    })

    const paramArray = Array.from(parameters)

    const groupedData: Record<string, ChartData[]> = {}
    paramArray.forEach((param) => {
        groupedData[param] = items
            .filter((item: any) => item.parameters?.[param]?.forecast_value != null)
            .map((item: any) => {
                const dateObj = new Date(item.forecast_date)
                return {
                    date: dateObj.toLocaleDateString("id-ID", { month: "short", day: "numeric" }),
                    fullDate: dateObj.toLocaleDateString("id-ID", { 
                        weekday: "long", 
                        year: "numeric", 
                        month: "long", 
                        day: "numeric" 
                    }),
                    value: item.parameters[param].forecast_value,
                }
            })
            .sort((a: any, b: any) => new Date(a.fullDate).getTime() - new Date(b.fullDate).getTime())
    })

    if (paramArray.length === 0) return <EmptyState />

    return (
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
                    <ParamChart param={param} data={groupedData[param]} />
                </TabsContent>
            ))}
        </Tabs>
    )
}
