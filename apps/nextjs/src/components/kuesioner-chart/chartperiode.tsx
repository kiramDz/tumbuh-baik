"use client";

import { useQuery } from "@tanstack/react-query";
import { getAllKuesionerPeriode } from "@/lib/fetch/files.fetch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, PieChart, Pie, Cell, LabelList, RadialBarChart, RadialBar } from "recharts";
import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartConfig } from "@/components/ui/chart";
import { useState, useMemo } from "react";
import { Calendar, Sprout, Loader2, AlertCircle, TrendingUp, DollarSign, MapPin } from "lucide-react";

const COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#84cc16", "#f97316", "#ec4899", "#6366f1"];

const chartConfig = {
  count: {
    label: "Jumlah",
    color: "hsl(var(--chart-1))",
  },
  tanam: {
    label: "Tanam",
    color: "hsl(142, 76%, 36%)",
  },
  panen: {
    label: "Panen",
    color: "hsl(221, 83%, 53%)",
  },
  produktivitas: {
    label: "Produktivitas",
    color: "hsl(38, 92%, 50%)",
  },
  pendapatan: {
    label: "Pendapatan",
    color: "hsl(142, 76%, 36%)",
  },
  pengeluaran: {
    label: "Pengeluaran",
    color: "hsl(0, 84%, 60%)",
  },
  keuntungan: {
    label: "Keuntungan",
    color: "hsl(221, 83%, 53%)",
  },
} satisfies ChartConfig;

type ChartType = "periode-tanam" | "produktivitas" | "ekonomi" | "pengelolaan" | "jenis-lahan" | "regional";

export default function ChartPeriode() {
  const [selectedChart, setSelectedChart] = useState<ChartType>("periode-tanam");
  const [selectedKabupaten, setSelectedKabupaten] = useState<string>("all");

  const { data: periodeData, isLoading, error } = useQuery({
    queryKey: ["kuesioner-periode-data"],
    queryFn: getAllKuesionerPeriode,
  });

  // Debug logging
  useMemo(() => {
    if (periodeData && periodeData.length > 0) {
      console.log("🔍 Sample data:", periodeData[0]);
      console.log("📋 Has kab_kota?", periodeData[0]?.kab_kota ? "✅ Yes" : "❌ No");
      console.log("🗂️ All fields:", Object.keys(periodeData[0] || {}));
    }
  }, [periodeData]);

  const kabupatenList = useMemo(() => {
    if (!periodeData || periodeData.length === 0) return [];
    const kabupatenSet = new Set<string>();
    periodeData.forEach((item: any) => {
      const kab = item.kab_kota || item.kabupaten || item.kabupaten_kota;
      if (kab && typeof kab === 'string' && kab.trim() !== '') {
        kabupatenSet.add(kab);
      }
    });
    const list = Array.from(kabupatenSet).sort();
    console.log("🏘️ Kabupaten list:", list);
    return list;
  }, [periodeData]);

  const filteredData = useMemo(() => {
    if (!periodeData) return [];
    if (selectedKabupaten === "all") return periodeData;
    return periodeData.filter((item: any) => {
      const kab = item.kab_kota || item.kabupaten || item.kabupaten_kota;
      return kab === selectedKabupaten;
    });
  }, [periodeData, selectedKabupaten]);

  const processedData = useMemo(() => {
    if (!filteredData || filteredData.length === 0) return null;
    
    const responses = filteredData;
    
    // Periode Tanam
    const bulanTanamGroups: { [key: string]: number } = {};
    const bulanPanenGroups: { [key: string]: number } = {};
    
    responses.forEach((response: any) => {
      const bulanTanam = response.bulan_tanam;
      const bulanPanen = response.bulan_panen;
      
      if (bulanTanam && bulanTanam.trim() !== '') {
        bulanTanamGroups[bulanTanam] = (bulanTanamGroups[bulanTanam] || 0) + 1;
      }
      
      if (bulanPanen && bulanPanen.trim() !== '') {
        bulanPanenGroups[bulanPanen] = (bulanPanenGroups[bulanPanen] || 0) + 1;
      }
    });

    const dataTanam = Object.entries(bulanTanamGroups).map(([bulan, count]) => ({
      bulan,
      count,
      type: 'tanam',
      percentage: ((count / responses.length) * 100).toFixed(1),
    }));

    const dataPanen = Object.entries(bulanPanenGroups).map(([bulan, count]) => ({
      bulan,
      count,
      type: 'panen',
      percentage: ((count / responses.length) * 100).toFixed(1),
    }));

    // Produktivitas
    const produktivitasData = responses
      .filter((item: any) => 
        item.luas_panen_m2 > 0 && 
        item.gunca_kg > 0 && 
        typeof item.luas_panen_m2 === 'number' && 
        typeof item.gunca_kg === 'number'
      )
      .map((item: any) => ({
        id_petani: item.id_petani?.toString().slice(-6) || 'N/A',
        produktivitas: (item.gunca_kg / item.luas_panen_m2) * 10000,
        luas_tanam: item.luas_tanam_m2 || 0,
        luas_panen: item.luas_panen_m2 || 0,
        hasil_panen: item.gunca_kg || 0,
        frekuensi_tanam: item.jml_tanam_padi_1th || 0,
        jenis_lahan: item.jenis_lahan || 'N/A'
      }))
      .sort((a: any, b: any) => b.produktivitas - a.produktivitas)
      .slice(0, 20);

    // Ekonomi
    const ekonomiData = responses
      .filter((item: any) => 
        item.harga_jual_perkg > 0 && 
        item.pengeluaran_rp > 0 && 
        item.gunca_kg > 0 &&
        typeof item.harga_jual_perkg === 'number' &&
        typeof item.pengeluaran_rp === 'number' &&
        typeof item.gunca_kg === 'number'
      )
      .map((item: any) => {
        const pendapatan = (item.gunca_kg || 0) * (item.harga_jual_perkg || 0);
        const pengeluaran = item.pengeluaran_rp || 0;
        const keuntungan = pendapatan - pengeluaran;
        
        return {
          id_petani: item.id_petani?.toString().slice(-6) || 'N/A',
          pendapatan,
          pengeluaran,
          keuntungan,
          harga_jual: item.harga_jual_perkg || 0,
          percentage: responses.length > 0 ? ((keuntungan / pendapatan) * 100).toFixed(1) : '0',
        };
      })
      .filter((item: any) => item.pendapatan > 0)
      .sort((a: any, b: any) => b.keuntungan - a.keuntungan)
      .slice(0, 15);

    // Pengelolaan
    const pengelolaanGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      const status = response.status_pengelolaan;
      if (status && status.trim() !== '') {
        pengelolaanGroups[status] = (pengelolaanGroups[status] || 0) + 1;
      }
    });

    const pengelolaan = Object.entries(pengelolaanGroups).map(([status, count]) => ({
      status,
      count,
      percentage: ((count / responses.length) * 100).toFixed(1),
    }));

    // Jenis Lahan
    const jenisLahanGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      const jenis = response.jenis_lahan;
      if (jenis && jenis.trim() !== '') {
        jenisLahanGroups[jenis] = (jenisLahanGroups[jenis] || 0) + 1;
      }
    });

    const jenisLahan = Object.entries(jenisLahanGroups).map(([jenis, count]) => ({
      jenis,
      count,
      percentage: ((count / responses.length) * 100).toFixed(1),
    }));

    // Regional
    const regionalGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      const kab = response.kab_kota || response.kabupaten || response.kabupaten_kota;
      if (kab && typeof kab === 'string' && kab.trim() !== '') {
        regionalGroups[kab] = (regionalGroups[kab] || 0) + 1;
      }
    });

    const regional = Object.entries(regionalGroups)
      .map(([kabupaten, count]) => ({
        kabupaten,
        count,
        percentage: ((count / responses.length) * 100).toFixed(1),
      }))
      .sort((a, b) => b.count - a.count);

    // Statistics
    const validProduktivitas = responses.filter((r: any) => 
      r.luas_panen_m2 > 0 && 
      r.gunca_kg > 0 && 
      typeof r.luas_panen_m2 === 'number' && 
      typeof r.gunca_kg === 'number'
    );
    
    const averageProduktivitas = validProduktivitas.length > 0 
      ? validProduktivitas.reduce((sum: number, r: any) => sum + ((r.gunca_kg / r.luas_panen_m2) * 10000), 0) / validProduktivitas.length
      : 0;

    const averageKeuntungan = ekonomiData.length > 0 
      ? ekonomiData.reduce((sum: number, item: any) => sum + item.keuntungan, 0) / ekonomiData.length 
      : 0;

    const validLuasTanam = responses.filter((r: any) => typeof r.luas_tanam_m2 === 'number' && r.luas_tanam_m2 > 0);
    const averageLuasTanam = validLuasTanam.length > 0
      ? validLuasTanam.reduce((sum: number, r: any) => sum + r.luas_tanam_m2, 0) / validLuasTanam.length
      : 0;

    const validFrekuensi = responses.filter((r: any) => typeof r.jml_tanam_padi_1th === 'number' && r.jml_tanam_padi_1th > 0);
    const averageFrekuensiTanam = validFrekuensi.length > 0
      ? validFrekuensi.reduce((sum: number, r: any) => sum + r.jml_tanam_padi_1th, 0) / validFrekuensi.length
      : 0;

    return {
      dataTanam,
      dataPanen,
      produktivitasData,
      ekonomiData,
      pengelolaan,
      jenisLahan,
      regional,
      totalResponses: responses.length,
      averageProduktivitas,
      averageKeuntungan,
      averageLuasTanam,
      averageFrekuensiTanam,
    };
  }, [filteredData]);

  const renderChart = () => {
    if (!processedData) return null;

    switch (selectedChart) {
      case "periode-tanam":
        return (
          <div className="space-y-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Periode Tanam</CardTitle>
                <CardDescription className="text-xs">Distribusi bulan tanam</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer config={chartConfig} className="h-[280px] w-full">
                  <BarChart data={processedData.dataTanam} margin={{ bottom: 60 }}>
                    <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-muted/30" />
                    <XAxis 
                      dataKey="bulan" 
                      angle={-45}
                      textAnchor="end"
                      height={70}
                      tickLine={false} 
                      axisLine={false}
                      tick={{ fontSize: 10 }}
                    />
                    <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={35} />
                    <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                    <Bar dataKey="count" fill="hsl(142, 76%, 36%)" radius={[4, 4, 0, 0]}>
                      <LabelList position="top" offset={8} className="fill-foreground" fontSize={10} />
                    </Bar>
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Periode Panen</CardTitle>
                <CardDescription className="text-xs">Distribusi bulan panen</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer config={chartConfig} className="h-[280px] w-full">
                  <BarChart data={processedData.dataPanen} margin={{ bottom: 60 }}>
                    <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-muted/30" />
                    <XAxis 
                      dataKey="bulan" 
                      angle={-45}
                      textAnchor="end"
                      height={70}
                      tickLine={false} 
                      axisLine={false}
                      tick={{ fontSize: 10 }}
                    />
                    <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={35} />
                    <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                    <Bar dataKey="count" fill="hsl(221, 83%, 53%)" radius={[4, 4, 0, 0]}>
                      <LabelList position="top" offset={8} className="fill-foreground" fontSize={10} />
                    </Bar>
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>
          </div>
        );

      case "produktivitas":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Produktivitas Tertinggi</CardTitle>
              <CardDescription className="text-xs">Top 20 produktivitas (kg/ha)</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[400px] w-full">
                <BarChart 
                  data={processedData.produktivitasData} 
                  layout="vertical"
                  margin={{ left: 0, right: 40 }}
                >
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <YAxis 
                    dataKey="id_petani" 
                    type="category"
                    tickLine={false} 
                    axisLine={false}
                    width={50}
                    tick={{ fontSize: 9 }}
                  />
                  <XAxis type="number" hide />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="produktivitas" fill="hsl(38, 92%, 50%)" radius={[0, 4, 4, 0]}>
                    <LabelList position="right" offset={8} className="fill-foreground" fontSize={10} formatter={(value: number) => value.toFixed(1)} />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "ekonomi":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Analisis Ekonomi</CardTitle>
              <CardDescription className="text-xs">Top 10 perbandingan pendapatan vs pengeluaran</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[500px] w-full">
                <BarChart 
                  data={processedData.ekonomiData.slice(0, 10)} 
                  layout="vertical"
                  margin={{ left: 0, right: 80 }}
                >
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <YAxis 
                    dataKey="id_petani" 
                    type="category"
                    tickLine={false} 
                    axisLine={false}
                    width={50}
                    tick={{ fontSize: 9 }}
                  />
                  <XAxis type="number" hide />
                  <ChartTooltip 
                    cursor={false} 
                    content={<ChartTooltipContent 
                      formatter={(value) => `Rp ${Number(value).toLocaleString()}`} 
                    />} 
                  />
                  <Bar dataKey="pendapatan" fill="hsl(142, 76%, 36%)" radius={[0, 4, 4, 0]} name="Pendapatan" />
                  <Bar dataKey="pengeluaran" fill="hsl(0, 84%, 60%)" radius={[0, 4, 4, 0]} name="Pengeluaran" />
                  <Bar dataKey="keuntungan" fill="hsl(221, 83%, 53%)" radius={[0, 4, 4, 0]} name="Keuntungan">
                    <LabelList position="right" offset={8} className="fill-foreground" fontSize={9} formatter={(value: number) => `${(value/1000000).toFixed(1)}M`} />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "pengelolaan":
        const totalPengelolaan = processedData.pengelolaan.reduce((sum: number, item: any) => sum + item.count, 0);
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Status Pengelolaan</CardTitle>
              <CardDescription className="text-xs">Distribusi status pengelolaan lahan</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[350px] w-full">
                <PieChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <ChartTooltip 
                    cursor={false} 
                    content={<ChartTooltipContent hideLabel />} 
                  />
                  <Pie
                    data={processedData.pengelolaan}
                    dataKey="count"
                    nameKey="status"
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                  >
                    {processedData.pengelolaan.map((entry: any, index: number) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                    <LabelList
                      position="outside"
                      className="fill-foreground"
                      fontSize={11}
                      formatter={(value: number, entry: any) => `${entry.status}: ${((value/totalPengelolaan)*100).toFixed(0)}%`}
                    />
                  </Pie>
                </PieChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "jenis-lahan":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Jenis Lahan</CardTitle>
              <CardDescription className="text-xs">Distribusi jenis lahan</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[300px] w-full">
                <BarChart data={processedData.jenisLahan} margin={{ bottom: 60 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <XAxis 
                    dataKey="jenis" 
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    tickLine={false} 
                    axisLine={false}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={35} />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" fill="hsl(262, 83%, 58%)" radius={[4, 4, 0, 0]}>
                    <LabelList position="top" offset={8} className="fill-foreground" fontSize={10} />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "regional":
        if (!processedData.regional || processedData.regional.length === 0) {
          return (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Data kabupaten tidak tersedia. Field kab_kota tidak ditemukan dalam respons API.
              </AlertDescription>
            </Alert>
          );
        }
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Distribusi Regional</CardTitle>
              <CardDescription className="text-xs">Jumlah data per kabupaten/kota</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[400px] w-full">
                <BarChart data={processedData.regional} margin={{ bottom: 60 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <XAxis 
                    dataKey="kabupaten" 
                    angle={-45}
                    textAnchor="end"
                    height={100}
                    tickLine={false} 
                    axisLine={false}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={35} />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" fill="hsl(167, 80%, 42%)" radius={[4, 4, 0, 0]}>
                    <LabelList position="top" offset={8} className="fill-foreground" fontSize={10} />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      default:
        return null;
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <Skeleton className="h-9 w-9 rounded-md" />
                  <div className="space-y-2 flex-1">
                    <Skeleton className="h-3 w-20" />
                    <Skeleton className="h-5 w-12" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        <Card>
          <CardContent className="flex items-center justify-center h-[300px]">
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Memuat data...</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Terjadi kesalahan saat memuat data. Silakan coba lagi.
        </AlertDescription>
      </Alert>
    );
  }

  if (!processedData) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Tidak ada data periode tersedia
        </AlertDescription>
      </Alert>
    );
  }

  const getChartTitle = () => {
    const titles = {
      "periode-tanam": "Periode Tanam & Panen",
      "produktivitas": "Produktivitas Padi",
      "ekonomi": "Analisis Ekonomi",
      "pengelolaan": "Status Pengelolaan",
      "jenis-lahan": "Jenis Lahan",
      "regional": "Distribusi Regional",
    };
    return titles[selectedChart];
  };

  return (
    <div className="space-y-4">
      {/* Stats */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-md bg-emerald-500/10 p-2.5">
                <Calendar className="h-4 w-4 text-emerald-600" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Total Data</p>
                <p className="text-lg font-semibold tabular-nums">{processedData.totalResponses}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-md bg-blue-500/10 p-2.5">
                <Sprout className="h-4 w-4 text-blue-600" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Produktivitas</p>
                <p className="text-lg font-semibold tabular-nums">
                  {processedData.averageProduktivitas.toFixed(1)} <span className="text-xs text-muted-foreground font-normal">kg/ha</span>
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-md bg-amber-500/10 p-2.5">
                <DollarSign className="h-4 w-4 text-amber-600" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Keuntungan Rata-rata</p>
                <p className="text-lg font-semibold tabular-nums">
                  {(processedData.averageKeuntungan / 1000000).toFixed(1)}M
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-md bg-purple-500/10 p-2.5">
                <TrendingUp className="h-4 w-4 text-purple-600" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Frekuensi Tanam</p>
                <p className="text-lg font-semibold tabular-nums">
                  {processedData.averageFrekuensiTanam.toFixed(1)}x <span className="text-xs text-muted-foreground font-normal">/tahun</span>
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="min-w-[180px] space-y-1">
              <label className="text-[11px] text-muted-foreground uppercase tracking-wide">Jenis Analisis</label>
              <Select value={selectedChart} onValueChange={(value: ChartType) => setSelectedChart(value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="periode-tanam">Periode Tanam</SelectItem>
                  <SelectItem value="produktivitas">Produktivitas</SelectItem>
                  <SelectItem value="ekonomi">Ekonomi</SelectItem>
                  <SelectItem value="pengelolaan">Pengelolaan</SelectItem>
                  <SelectItem value="jenis-lahan">Jenis Lahan</SelectItem>
                  <SelectItem value="regional">Regional</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="min-w-[180px] space-y-1">
              <label className="text-[11px] text-muted-foreground uppercase tracking-wide">Kabupaten/Kota</label>
              <Select value={selectedKabupaten} onValueChange={setSelectedKabupaten}>
                <SelectTrigger>
                  <SelectValue placeholder="Semua Kabupaten" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Semua Kabupaten</SelectItem>
                  {kabupatenList.map((kab) => (
                    <SelectItem key={kab} value={kab}>{kab}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="ml-auto text-xs text-muted-foreground">
              <span className="font-medium text-foreground">{processedData.totalResponses}</span> data ditampilkan
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Chart */}
      {renderChart()}
    </div>
  );
}