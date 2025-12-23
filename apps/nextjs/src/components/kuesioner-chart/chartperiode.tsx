"use client";

import { useQuery } from "@tanstack/react-query";
import { getAllKuesionerPeriode } from "@/lib/fetch/files.fetch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Legend, AreaChart, Area } from "recharts";
import { useState, useMemo } from "react";
import { BarChart3, Calendar, Sprout, MapPin, Loader2, AlertCircle } from "lucide-react";

const COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#84cc16", "#f97316", "#ec4899", "#6366f1"];

type ChartType = "periode-tanam" | "produktivitas" | "ekonomi" | "pengelolaan" | "jenis-lahan" | "regional";

export default function ChartPeriode() {
  const [selectedChart, setSelectedChart] = useState<ChartType>("periode-tanam");
  const [selectedKabupaten, setSelectedKabupaten] = useState<string>("all");

  const { data: periodeData, isLoading, error } = useQuery({
    queryKey: ["kuesioner-periode-data"],
    queryFn: getAllKuesionerPeriode,
  });

  const kabupatenList = useMemo(() => {
    if (!periodeData) return [];
    const uniqueKabupaten = [...new Set(periodeData.map((item: any) => item.kab_kota))];
    return uniqueKabupaten.filter(Boolean).sort();
  }, [periodeData]);

  const filteredData = useMemo(() => {
    if (!periodeData) return [];
    if (selectedKabupaten === "all") return periodeData;
    return periodeData.filter((item: any) => item.kab_kota === selectedKabupaten);
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
    const regionalGroups: { [key: string]: { 
      count: number, 
      total_luas: number, 
      total_produksi: number,
      total_pengeluaran: number,
      total_pendapatan: number
    } } = {};
    
    responses.forEach((response: any) => {
      const region = response.kab_kota;
      if (region && region.trim() !== '') {
        if (!regionalGroups[region]) {
          regionalGroups[region] = { 
            count: 0, 
            total_luas: 0, 
            total_produksi: 0,
            total_pengeluaran: 0,
            total_pendapatan: 0
          };
        }
        regionalGroups[region].count++;
        regionalGroups[region].total_luas += response.luas_tanam_m2 || 0;
        regionalGroups[region].total_produksi += response.gunca_kg || 0;
        regionalGroups[region].total_pengeluaran += response.pengeluaran_rp || 0;
        regionalGroups[region].total_pendapatan += ((response.gunca_kg || 0) * (response.harga_jual_perkg || 0));
      }
    });

    const regional = Object.entries(regionalGroups).map(([region, data]) => ({
      region,
      count: data.count,
      total_luas: data.total_luas,
      total_produksi: data.total_produksi,
      rata_produktivitas: data.total_luas > 0 ? (data.total_produksi / data.total_luas) * 10000 : 0,
      rata_pendapatan: data.count > 0 ? data.total_pendapatan / data.count : 0,
      percentage: ((data.count / responses.length) * 100).toFixed(1),
    }));

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

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <Card className="border shadow-md">
          <CardContent className="p-3">
            <p className="font-medium text-sm mb-1">
              {payload[0].payload.bulan || payload[0].payload.id_petani || payload[0].payload.status || payload[0].payload.jenis || payload[0].payload.region}
            </p>
            <div className="space-y-1">
              {payload.map((entry: any, index: number) => (
                <div key={index} className="flex items-center gap-2 text-xs">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
                  <span>
                    {entry.name}: {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}
                  </span>
                  {entry.payload?.percentage && (
                    <span className="text-muted-foreground">({entry.payload.percentage}%)</span>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      );
    }
    return null;
  };

  const renderChart = () => {
    if (!processedData) return null;

    switch (selectedChart) {
      case "periode-tanam":
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-medium mb-3 text-muted-foreground">Periode Tanam</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={processedData.dataTanam} margin={{ bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="bulan" angle={-35} textAnchor="end" height={60} className="text-xs" />
                  <YAxis className="text-xs" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} name="Jumlah" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h3 className="text-sm font-medium mb-3 text-muted-foreground">Periode Panen</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={processedData.dataPanen} margin={{ bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="bulan" angle={-35} textAnchor="end" height={60} className="text-xs" />
                  <YAxis className="text-xs" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Jumlah" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        );

      case "produktivitas":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={processedData.produktivitasData} margin={{ bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="id_petani" angle={-35} textAnchor="end" height={70} className="text-xs" />
              <YAxis className="text-xs" />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="produktivitas" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.3} name="Produktivitas (kg/ha)" />
            </AreaChart>
          </ResponsiveContainer>
        );

      case "ekonomi":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={processedData.ekonomiData} margin={{ bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="id_petani" angle={-35} textAnchor="end" height={70} className="text-xs" />
              <YAxis className="text-xs" tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`} />
              <Tooltip content={<CustomTooltip />} formatter={(value) => `Rp ${value.toLocaleString()}`} />
              <Legend />
              <Bar dataKey="pendapatan" fill="#10b981" name="Pendapatan" radius={[2, 2, 0, 0]} />
              <Bar dataKey="pengeluaran" fill="#ef4444" name="Pengeluaran" radius={[2, 2, 0, 0]} />
              <Bar dataKey="keuntungan" fill="#3b82f6" name="Keuntungan" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        );

      case "pengelolaan":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={processedData.pengelolaan}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ status, percentage }) => `${status}: ${percentage}%`}
                outerRadius={130}
                innerRadius={50}
                dataKey="count"
              >
                {processedData.pengelolaan.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );

      case "jenis-lahan":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={processedData.jenisLahan} margin={{ bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="jenis" angle={-35} textAnchor="end" height={70} className="text-xs" />
              <YAxis className="text-xs" />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Jumlah" />
            </BarChart>
          </ResponsiveContainer>
        );

      case "regional":
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={processedData.regional} margin={{ bottom: 80 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="region" angle={-45} textAnchor="end" height={80} className="text-xs" />
              <YAxis className="text-xs" />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} name="Jumlah" />
            </BarChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-64 mt-2" />
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-center h-[400px]">
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
                <p className="text-sm text-muted-foreground">Memuat data periode...</p>
              </div>
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
          Tidak ada data tersedia untuk {selectedKabupaten === "all" ? "semua kabupaten" : selectedKabupaten}
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
    <div className="space-y-6">
      {/* Filters Card */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Filter Data</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-2">
                <MapPin className="w-4 h-4 text-muted-foreground" />
                Kabupaten
              </label>
              <Select value={selectedKabupaten} onValueChange={setSelectedKabupaten}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Semua Kabupaten</SelectItem>
                  {kabupatenList.map((kab) => (
                    <SelectItem key={kab as string} value={kab as string}>
                      {kab as string}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-muted-foreground" />
                Jenis Analisis
              </label>
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
          </div>
        </CardContent>
      </Card>

      {/* Chart Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">{getChartTitle()}</CardTitle>
              <CardDescription className="mt-1">
                {selectedKabupaten === "all" ? "Semua kabupaten" : selectedKabupaten}
              </CardDescription>
            </div>
            <Badge variant="secondary">
              {processedData.totalResponses} responden
            </Badge>
          </div>
        </CardHeader>
        <Separator />
        <CardContent className="pt-6">
          {renderChart()}
        </CardContent>
      </Card>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="border-l-4 border-l-green-500">
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Total Data</p>
                <p className="text-2xl font-bold text-green-600">{processedData.totalResponses}</p>
              </div>
              <Calendar className="w-8 h-8 text-green-500 opacity-50" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-blue-500">
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Produktivitas</p>
                <p className="text-2xl font-bold text-blue-600">
                  {processedData.averageProduktivitas.toFixed(1)}
                </p>
                <p className="text-xs text-muted-foreground">kg/ha</p>
              </div>
              <Sprout className="w-8 h-8 text-blue-500 opacity-50" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-amber-500">
          <CardContent className="pt-4">
            <div>
              <p className="text-sm text-muted-foreground mb-1">Keuntungan</p>
              <p className="text-2xl font-bold text-amber-600">
                {(processedData.averageKeuntungan / 1000000).toFixed(1)}M
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-purple-500">
          <CardContent className="pt-4">
            <div>
              <p className="text-sm text-muted-foreground mb-1">Frekuensi</p>
              <p className="text-2xl font-bold text-purple-600">
                {processedData.averageFrekuensiTanam.toFixed(1)}x
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}