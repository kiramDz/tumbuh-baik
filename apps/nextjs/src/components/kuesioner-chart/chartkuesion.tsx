"use client";

import { useQuery } from "@tanstack/react-query";
import { getAllKuesionerPetani } from "@/lib/fetch/files.fetch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { 
  Bar, 
  BarChart, 
  Pie, 
  PieChart, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  LabelList,
  RadialBar,
  RadialBarChart,
  PolarAngleAxis,
  Cell,
  Label
} from "recharts";
import { useState, useMemo } from "react";
import { Users, MapPin, Loader2, AlertCircle, Wheat, Calendar, LandPlot } from "lucide-react";

type ChartType = "demographics" | "education" | "farming-experience" | "land-ownership" | "varieties" | "regional";

const chartConfig = {
  count: {
    label: "Jumlah",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig;

const COLORS = {
  primary: "hsl(var(--chart-1))",
  secondary: "hsl(var(--chart-2))",
  tertiary: "hsl(var(--chart-3))",
  quaternary: "hsl(var(--chart-4))",
  quinary: "hsl(var(--chart-5))",
};

const PIE_COLORS = [
  "hsl(142, 76%, 36%)",   // green
  "hsl(346, 77%, 50%)",   // rose
  "hsl(221, 83%, 53%)",   // blue
  "hsl(38, 92%, 50%)",    // amber
  "hsl(262, 83%, 58%)",   // violet
  "hsl(173, 80%, 40%)",   // teal
  "hsl(24, 95%, 53%)",    // orange
  "hsl(280, 65%, 60%)",   // purple
];

export default function ChartKuesioner() {
  const [selectedChart, setSelectedChart] = useState<ChartType>("demographics");
  const [selectedKabupaten, setSelectedKabupaten] = useState<string>("all");

  const { data: kuisionerData, isLoading, error } = useQuery({
    queryKey: ["kuesioner-data"],
    queryFn: getAllKuesionerPetani,
  });

  const kabupatenList = useMemo(() => {
    if (!kuisionerData) return [];
    const uniqueKabupaten = [...new Set(kuisionerData.map((item: any) => item.kab_kota))];
    return uniqueKabupaten.filter(Boolean).sort();
  }, [kuisionerData]);

  const filteredData = useMemo(() => {
    if (!kuisionerData) return [];
    if (selectedKabupaten === "all") return kuisionerData;
    return kuisionerData.filter((item: any) => item.kab_kota === selectedKabupaten);
  }, [kuisionerData, selectedKabupaten]);

  const processedData = useMemo(() => {
    if (!filteredData || filteredData.length === 0) return null;
    
    const responses = filteredData;
    const currentYear = new Date().getFullYear();
    
    // Age groups
    const ageGroups = { "18-30": 0, "31-40": 0, "41-50": 0, "51-60": 0, "60+": 0 };
    const genderGroups = { "Laki-laki": 0, "Perempuan": 0 };
    
    responses.forEach((response: any) => {
      const age = response.umur;
      if (age >= 18 && age <= 30) ageGroups["18-30"]++;
      else if (age >= 31 && age <= 40) ageGroups["31-40"]++;
      else if (age >= 41 && age <= 50) ageGroups["41-50"]++;
      else if (age >= 51 && age <= 60) ageGroups["51-60"]++;
      else if (age > 60) ageGroups["60+"]++;

      if (response.jenis_kelamin === "Laki-laki" || response.jenis_kelamin === "L") {
        genderGroups["Laki-laki"]++;
      } else {
        genderGroups["Perempuan"]++;
      }
    });

    const demographics = Object.entries(ageGroups).map(([category, count], index) => ({
      category: `${category} tahun`,
      count,
      fill: PIE_COLORS[index % PIE_COLORS.length],
    }));

    const genderData = Object.entries(genderGroups).map(([gender, count], index) => ({
      gender,
      count,
      fill: index === 0 ? "hsl(221, 83%, 53%)" : "hsl(346, 77%, 50%)",
    }));

    const totalGender = genderGroups["Laki-laki"] + genderGroups["Perempuan"];

    // Education
    const educationGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      const education = response.pendidikan_terakhir;
      if (education && education.trim() !== '') {
        educationGroups[education.trim()] = (educationGroups[education.trim()] || 0) + 1;
      }
    });

    const education = Object.entries(educationGroups)
      .sort((a, b) => b[1] - a[1])
      .map(([level, count], index) => ({
        level,
        count,
        fill: PIE_COLORS[index % PIE_COLORS.length],
      }));

    // Farming experience
    const experienceGroups = { "1-5": 0, "6-10": 0, "11-20": 0, "20+": 0 };

    responses.forEach((response: any) => {
      const startYear = response.tahun_mulai_bertani;
      if (startYear && startYear > 1900) {
        const experience = currentYear - startYear;
        if (experience >= 1 && experience <= 5) experienceGroups["1-5"]++;
        else if (experience >= 6 && experience <= 10) experienceGroups["6-10"]++;
        else if (experience >= 11 && experience <= 20) experienceGroups["11-20"]++;
        else if (experience > 20) experienceGroups["20+"]++;
      }
    });

    const farmingExperience = Object.entries(experienceGroups).map(([range, count], index) => ({
      range: `${range} tahun`,
      count,
      fill: PIE_COLORS[index % PIE_COLORS.length],
    }));

    // Land ownership for radial chart
    const landOwnershipGroups = { "<1.000": 0, "1.000-5.000": 0, "5.000-10.000": 0, ">10.000": 0 };

    responses.forEach((response: any) => {
      const totalLand = response.total_lahan_m2;
      if (totalLand && totalLand > 0) {
        if (totalLand < 1000) landOwnershipGroups["<1.000"]++;
        else if (totalLand >= 1000 && totalLand < 5000) landOwnershipGroups["1.000-5.000"]++;
        else if (totalLand >= 5000 && totalLand < 10000) landOwnershipGroups["5.000-10.000"]++;
        else if (totalLand >= 10000) landOwnershipGroups[">10.000"]++;
      }
    });

    const maxLandCount = Math.max(...Object.values(landOwnershipGroups));
    const landOwnership = Object.entries(landOwnershipGroups).map(([range, count], index) => ({
      range: `${range} m²`,
      count,
      fill: PIE_COLORS[index % PIE_COLORS.length],
    }));

    // Rice varieties
    const varietiesGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      const variety = response.varietas_padi;
      if (variety) {
        let varietyName = '';
        if (typeof variety === 'string') {
          varietyName = variety.trim();
        } else if (typeof variety === 'object' && variety !== null) {
          varietyName = variety.nama || variety.name || variety.variety || String(variety);
        } else {
          varietyName = String(variety).trim();
        }
        
        if (varietyName && varietyName !== '' && varietyName !== 'null' && varietyName !== 'undefined') {
          varietiesGroups[varietyName] = (varietiesGroups[varietyName] || 0) + 1;
        }
      }
    });

    const varieties = Object.entries(varietiesGroups)
      .map(([variety, count], index) => ({
        variety,
        count,
        fill: PIE_COLORS[index % PIE_COLORS.length],
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    // Regional distribution
    const regionalGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      const region = selectedKabupaten === "all" ? response.kab_kota : response.kecamatan;
      if (region && region.trim() !== '') {
        regionalGroups[region.trim()] = (regionalGroups[region.trim()] || 0) + 1;
      }
    });

    const regional = Object.entries(regionalGroups)
      .map(([region, count], index) => ({
        region,
        count,
        fill: PIE_COLORS[index % PIE_COLORS.length],
      }))
      .sort((a, b) => b.count - a.count);

    return {
      demographics,
      genderData,
      totalGender,
      education,
      farmingExperience,
      landOwnership,
      maxLandCount,
      varieties,
      regional,
      totalResponses: responses.length,
      averageAge: responses.reduce((sum: number, r: any) => sum + (r.umur || 0), 0) / responses.length,
      averageExperience: responses.reduce((sum: number, r: any) => {
        const startYear = r.tahun_mulai_bertani;
        return sum + (startYear ? currentYear - startYear : 0);
      }, 0) / responses.length,
      averageLandSize: responses.reduce((sum: number, r: any) => sum + (r.total_lahan_m2 || 0), 0) / responses.length,
    };
  }, [filteredData, selectedKabupaten]);

  const renderChart = () => {
    if (!processedData) return null;

    switch (selectedChart) {
      case "demographics":
        return (
          <div className="grid gap-4 md:grid-cols-2">
            {/* Age - Horizontal Bar */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Distribusi Usia</CardTitle>
                <CardDescription className="text-xs">Kelompok usia responden</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer config={chartConfig} className="h-[240px] w-full">
                  <BarChart 
                    data={processedData.demographics} 
                    layout="vertical"
                    margin={{ left: 0, right: 32 }}
                  >
                    <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted/30" />
                    <YAxis 
                      dataKey="category" 
                      type="category"
                      tickLine={false} 
                      axisLine={false}
                      width={70}
                      tick={{ fontSize: 11 }}
                    />
                    <XAxis type="number" hide />
                    <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                    <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                      {processedData.demographics.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                      <LabelList dataKey="count" position="right" className="fill-foreground text-xs font-medium" />
                    </Bar>
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>

            {/* Gender - Pie with Center Label */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Jenis Kelamin</CardTitle>
                <CardDescription className="text-xs">Proporsi gender responden</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer config={chartConfig} className="mx-auto h-[200px] w-full">
                  <PieChart>
                    <ChartTooltip content={<ChartTooltipContent nameKey="gender" />} />
                    <Pie
                      data={processedData.genderData}
                      dataKey="count"
                      nameKey="gender"
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={75}
                      strokeWidth={3}
                      stroke="hsl(var(--background))"
                    >
                      {processedData.genderData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                      <Label
                        content={({ viewBox }) => {
                          if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                            return (
                              <text x={viewBox.cx} y={viewBox.cy} textAnchor="middle" dominantBaseline="middle">
                                <tspan x={viewBox.cx} y={viewBox.cy} className="fill-foreground text-2xl font-bold">
                                  {processedData.totalGender}
                                </tspan>
                                <tspan x={viewBox.cx} y={(viewBox.cy || 0) + 18} className="fill-muted-foreground text-xs">
                                  Total
                                </tspan>
                              </text>
                            )
                          }
                        }}
                      />
                    </Pie>
                  </PieChart>
                </ChartContainer>
                <div className="flex justify-center gap-6 mt-2">
                  {processedData.genderData.map((item) => (
                    <div key={item.gender} className="flex items-center gap-2 text-sm">
                      <div className="h-3 w-3 rounded-full" style={{ backgroundColor: item.fill }} />
                      <span className="text-muted-foreground">{item.gender}</span>
                      <span className="font-semibold">{item.count}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        );

      case "education":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Tingkat Pendidikan</CardTitle>
              <CardDescription className="text-xs">Pendidikan terakhir responden</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[280px] w-full">
                <BarChart 
                  data={processedData.education} 
                  layout="vertical"
                  margin={{ left: 0, right: 40 }}
                >
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <YAxis 
                    dataKey="level" 
                    type="category"
                    tickLine={false} 
                    axisLine={false}
                    width={80}
                    tick={{ fontSize: 11 }}
                  />
                  <XAxis type="number" hide />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {processedData.education.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="count" position="right" className="fill-foreground text-xs font-medium" />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "farming-experience":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Pengalaman Bertani</CardTitle>
              <CardDescription className="text-xs">
                Rata-rata: {processedData.averageExperience.toFixed(0)} tahun pengalaman
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[280px] w-full">
                <BarChart data={processedData.farmingExperience} margin={{ top: 20 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <XAxis 
                    dataKey="range" 
                    tickLine={false} 
                    axisLine={false}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={35} />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {processedData.farmingExperience.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="count" position="top" className="fill-foreground text-xs font-medium" />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "land-ownership":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Kepemilikan Lahan</CardTitle>
              <CardDescription className="text-xs">
                Rata-rata luas: {processedData.averageLandSize.toLocaleString('id-ID', { maximumFractionDigits: 0 })} m²
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <ChartContainer config={chartConfig} className="mx-auto h-[220px] w-full">
                  <RadialBarChart
                    data={processedData.landOwnership}
                    innerRadius={30}
                    outerRadius={100}
                    startAngle={180}
                    endAngle={0}
                  >
                    <ChartTooltip cursor={false} content={<ChartTooltipContent nameKey="range" />} />
                    <PolarAngleAxis type="number" domain={[0, processedData.maxLandCount]} tick={false} />
                    <RadialBar
                      dataKey="count"
                      background={{ fill: "hsl(var(--muted))" }}
                      cornerRadius={4}
                    >
                      {processedData.landOwnership.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </RadialBar>
                  </RadialBarChart>
                </ChartContainer>
                <div className="flex flex-col justify-center gap-3">
                  {processedData.landOwnership.map((item) => (
                    <div key={item.range} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: item.fill }} />
                        <span className="text-sm text-muted-foreground">{item.range}</span>
                      </div>
                      <span className="text-sm font-semibold tabular-nums">{item.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        );

      case "varieties":
        if (!processedData.varieties || processedData.varieties.length === 0) {
          return (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>Tidak ada data varietas padi</AlertDescription>
            </Alert>
          );
        }
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Varietas Padi</CardTitle>
              <CardDescription className="text-xs">Top 10 varietas yang ditanam</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[350px] w-full">
                <BarChart 
                  data={processedData.varieties} 
                  layout="vertical"
                  margin={{ left: 0, right: 40 }}
                >
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <YAxis 
                    dataKey="variety" 
                    type="category"
                    tickLine={false} 
                    axisLine={false}
                    width={100}
                    tick={{ fontSize: 10 }}
                    tickFormatter={(value) => value.length > 14 ? value.slice(0, 14) + '...' : value}
                  />
                  <XAxis type="number" hide />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {processedData.varieties.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="count" position="right" className="fill-foreground text-xs font-medium" />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "regional":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">
                Distribusi {selectedKabupaten === "all" ? "Kabupaten" : "Kecamatan"}
              </CardTitle>
              <CardDescription className="text-xs">Sebaran lokasi responden</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer 
                config={chartConfig} 
                className="w-full"
                style={{ height: Math.max(250, processedData.regional.length * 28) }}
              >
                <BarChart 
                  data={processedData.regional} 
                  layout="vertical"
                  margin={{ left: 0, right: 40 }}
                >
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <YAxis 
                    dataKey="region" 
                    type="category"
                    tickLine={false} 
                    axisLine={false}
                    width={120}
                    tick={{ fontSize: 10 }}
                    tickFormatter={(value) => value.length > 18 ? value.slice(0, 18) + '...' : value}
                  />
                  <XAxis type="number" hide />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {processedData.regional.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="count" position="right" className="fill-foreground text-xs font-medium" />
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
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <Skeleton className="h-9 w-9 rounded-md" />
                  <div className="space-y-1.5">
                    <Skeleton className="h-3 w-16" />
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

  if (error || !processedData) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>Gagal memuat data. Silakan muat ulang halaman.</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      {/* Stats */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-md bg-emerald-500/10 p-2.5">
                <Users className="h-4 w-4 text-emerald-600" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Total Petani</p>
                <p className="text-lg font-semibold tabular-nums">{processedData.totalResponses}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-md bg-blue-500/10 p-2.5">
                <Calendar className="h-4 w-4 text-blue-600" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Rata-rata Usia</p>
                <p className="text-lg font-semibold tabular-nums">{processedData.averageAge.toFixed(0)} thn</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-md bg-amber-500/10 p-2.5">
                <Wheat className="h-4 w-4 text-amber-600" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Pengalaman</p>
                <p className="text-lg font-semibold tabular-nums">{processedData.averageExperience.toFixed(0)} thn</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-md bg-purple-500/10 p-2.5">
                <LandPlot className="h-4 w-4 text-purple-600" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Luas Lahan</p>
                <p className="text-lg font-semibold tabular-nums">
                  {processedData.averageLandSize >= 1000 
                    ? `${(processedData.averageLandSize / 1000).toFixed(1)}k` 
                    : processedData.averageLandSize.toFixed(0)
                  } m²
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
              <label className="text-[11px] text-muted-foreground uppercase tracking-wide flex items-center gap-1">
                <MapPin className="h-3 w-3" />
                Wilayah
              </label>
              <Select value={selectedKabupaten} onValueChange={setSelectedKabupaten}>
                <SelectTrigger className="h-8 text-sm">
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

            <div className="min-w-[180px] space-y-1">
              <label className="text-[11px] text-muted-foreground uppercase tracking-wide">Jenis Analisis</label>
              <Select value={selectedChart} onValueChange={(v: ChartType) => setSelectedChart(v)}>
                <SelectTrigger className="h-8 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="demographics">Demografi</SelectItem>
                  <SelectItem value="education">Pendidikan</SelectItem>
                  <SelectItem value="farming-experience">Pengalaman Bertani</SelectItem>
                  <SelectItem value="land-ownership">Kepemilikan Lahan</SelectItem>
                  <SelectItem value="varieties">Varietas Padi</SelectItem>
                  <SelectItem value="regional">Distribusi Wilayah</SelectItem>
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