"use client";

import { useQuery } from "@tanstack/react-query";
import { getAllKuesionerManajemen } from "@/lib/fetch/files.fetch";
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
  Cell,
  Label,
  RadialBar,
  RadialBarChart,
  PolarAngleAxis
} from "recharts";
import { useState, useMemo } from "react";
import { Users, MapPin, Tractor, Wifi, Loader2, AlertCircle, Droplets, Sprout, BarChart3 } from "lucide-react";

type ManagementChartType = "technology" | "irrigation" | "spraying" | "harvest" | "information" | "membership" | "climate-awareness" | "failure-analysis" | "penyebab-gagal" | "teknologi-lain" | "respon-pergeseran" | "overall-adoption" | "regional";

const chartConfig = {
  ya: {
    label: "Ya",
    color: "hsl(var(--chart-1))",
  },
  tidak: {
    label: "Tidak",
    color: "hsl(var(--chart-2))",
  },
  count: {
    label: "Jumlah",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig;

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

export default function ChartManajemen() {
  const [selectedChart, setSelectedChart] = useState<ManagementChartType>("technology");
  const [selectedKabupaten, setSelectedKabupaten] = useState<string>("all");

  const { data: manajemenData, isLoading, error } = useQuery({
    queryKey: ["manajemen-data"],
    queryFn: getAllKuesionerManajemen,
  });

  const kabupatenList = useMemo(() => {
    if (!manajemenData) return [];
    const uniqueKabupaten = [...new Set(manajemenData.map((item: any) => item.kab_kota))];
    return uniqueKabupaten.filter(Boolean).sort();
  }, [manajemenData]);

  const filteredData = useMemo(() => {
    if (!manajemenData) return [];
    if (selectedKabupaten === "all") return manajemenData;
    return manajemenData.filter((item: any) => item.kab_kota === selectedKabupaten);
  }, [manajemenData, selectedKabupaten]);

  const processedData = useMemo(() => {
    if (!filteredData || filteredData.length === 0) return null;
    
    const responses = filteredData;
    
    // Technology Adoption
    const technologyData = [
      {
        category: "Pembajakan Modern",
        ya: responses.filter((r: any) => r.pembajakan_lahan_modern === 'Ya' || r.pembajakan_lahan_modern === 'ya').length,
        tidak: responses.filter((r: any) => r.pembajakan_lahan_modern === 'Tidak' || r.pembajakan_lahan_modern === 'tidak').length,
      },
      {
        category: "Mesin Potong",
        ya: responses.filter((r: any) => r.panen_mesin_potong === 'Ya' || r.panen_mesin_potong === 'ya').length,
        tidak: responses.filter((r: any) => r.panen_mesin_potong === 'Tidak' || r.panen_mesin_potong === 'tidak').length,
      },
      {
        category: "Internet",
        ya: responses.filter((r: any) => r.pakai_internet === 'Ya' || r.pakai_internet === 'ya').length,
        tidak: responses.filter((r: any) => r.pakai_internet === 'Tidak' || r.pakai_internet === 'tidak').length,
      },
    ].map((item, index) => ({
      ...item,
      fill: PIE_COLORS[index % PIE_COLORS.length],
      percentage: responses.length > 0 ? ((item.ya / responses.length) * 100).toFixed(1) : '0',
    }));

    // Irrigation Technology
    const irrigationData = [
      {
        method: "Sumur Bor",
        count: responses.filter((r: any) => r.pengairan_sumur_bor === 'Ya' || r.pengairan_sumur_bor === 'ya').length,
      },
      {
        method: "Pompa Air",
        count: responses.filter((r: any) => r.pengairan_pompa_air === 'Ya' || r.pengairan_pompa_air === 'ya').length,
      },
    ].map((item, index) => ({
      ...item,
      fill: PIE_COLORS[index % PIE_COLORS.length],
      percentage: responses.length > 0 ? ((item.count / responses.length) * 100).toFixed(1) : '0',
    }));

    // Spraying Technology
    const sprayingData = [
      {
        type: "Pompa Tangan",
        count: responses.filter((r: any) => r.penyemprotan_pompa_tangan === 'Ya' || r.penyemprotan_pompa_tangan === 'ya').length,
      },
      {
        type: "Pompa Elektrik",
        count: responses.filter((r: any) => r.penyemprotan_pompa_elektrik === 'Ya' || r.penyemprotan_pompa_elektrik === 'ya').length,
      },
    ].map((item, index) => ({
      ...item,
      fill: PIE_COLORS[index % PIE_COLORS.length],
      percentage: responses.length > 0 ? ((item.count / responses.length) * 100).toFixed(1) : '0',
    }));

    // Harvest Methods
    const harvestYa = responses.filter((r: any) => r.panen_mesin_potong === 'Ya' || r.panen_mesin_potong === 'ya').length;
    const harvestTidak = responses.filter((r: any) => r.panen_mesin_potong === 'Tidak' || r.panen_mesin_potong === 'tidak').length;
    
    const harvestData = [
      { method: "Mesin Modern", count: harvestYa, fill: PIE_COLORS[0] },
      { method: "Manual", count: harvestTidak, fill: PIE_COLORS[1] },
    ];
    const totalHarvest = harvestYa + harvestTidak;

    // Information Sources
    const informationData = [
      {
        source: "Penyuluh",
        count: responses.filter((r: any) => r.info_penyuluh === 'Ya' || r.info_penyuluh === 'ya').length,
      },
      {
        source: "Keuchik",
        count: responses.filter((r: any) => r.info_keuchik === 'Ya' || r.info_keuchik === 'ya').length,
      },
      {
        source: "Keujrun Blang",
        count: responses.filter((r: any) => r.info_keujrun_blang === 'Ya' || r.info_keujrun_blang === 'ya').length,
      },
    ].map((item, index) => ({
      ...item,
      fill: PIE_COLORS[index % PIE_COLORS.length],
      percentage: responses.length > 0 ? ((item.count / responses.length) * 100).toFixed(1) : '0',
    }));

    // Membership
    const membershipYa = responses.filter((r: any) => r.anggota_kelompok_tani === 'Ya' || r.anggota_kelompok_tani === 'ya').length;
    const membershipTidak = responses.filter((r: any) => r.anggota_kelompok_tani === 'Tidak' || r.anggota_kelompok_tani === 'tidak').length;
    
    const membershipData = [
      { status: "Anggota", count: membershipYa, fill: PIE_COLORS[0] },
      { status: "Bukan Anggota", count: membershipTidak, fill: PIE_COLORS[1] },
    ];
    const totalMembership = membershipYa + membershipTidak;

    // Climate Awareness
    const climateAwarenessData = [
      {
        awareness: "Tahu KATAM",
        count: responses.filter((r: any) => r.tahu_katam === 'Ya' || r.tahu_katam === 'ya').length,
      },
      {
        awareness: "Tahu Pergeseran",
        count: responses.filter((r: any) => r.tahu_pergeseran_musim === 'Ya' || r.tahu_pergeseran_musim === 'ya').length,
      },
    ].map((item, index) => ({
      ...item,
      fill: PIE_COLORS[index % PIE_COLORS.length],
      percentage: responses.length > 0 ? ((item.count / responses.length) * 100).toFixed(1) : '0',
    }));

    // Respon Pergeseran Musim
    const responPergeseranGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      if (response.respon_pergeseran_musim && response.respon_pergeseran_musim.trim() !== '' && response.respon_pergeseran_musim !== 'Tidak Ada' && response.respon_pergeseran_musim !== '-') {
        const respon = response.respon_pergeseran_musim.trim();
        responPergeseranGroups[respon] = (responPergeseranGroups[respon] || 0) + 1;
      }
    });

    const responPergeseranData = Object.entries(responPergeseranGroups)
      .map(([respon, count], index) => ({
        respon: respon.length > 25 ? respon.slice(0, 25) + '...' : respon,
        fullRespon: respon,
        count,
        fill: PIE_COLORS[index % PIE_COLORS.length],
        percentage: responses.length > 0 ? ((count / responses.length) * 100).toFixed(1) : '0',
      }))
      .sort((a, b) => b.count - a.count);

    // Failure Analysis
    const failureYa = responses.filter((r: any) => r.pernah_gagal_tanam === 'Ya' || r.pernah_gagal_tanam === 'ya').length;
    const failureTidak = responses.filter((r: any) => r.pernah_gagal_tanam === 'Tidak' || r.pernah_gagal_tanam === 'tidak').length;
    const totalFailure = failureYa + failureTidak;

    const failureAnalysis = [
      { status: "Pernah Gagal", count: failureYa, fill: "hsl(0, 84%, 60%)" },
      { status: "Tidak Pernah", count: failureTidak, fill: "hsl(142, 76%, 36%)" },
    ];

    // Penyebab Gagal
    const penyebabGagalGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      if (response.penyebab_gagal && response.penyebab_gagal.trim() !== '' && response.penyebab_gagal !== 'Tidak Ada') {
        const penyebab = response.penyebab_gagal.trim();
        penyebabGagalGroups[penyebab] = (penyebabGagalGroups[penyebab] || 0) + 1;
      }
    });

    const penyebabGagalData = Object.entries(penyebabGagalGroups)
      .map(([penyebab, count], index) => ({
        penyebab: penyebab.length > 20 ? penyebab.slice(0, 20) + '...' : penyebab,
        fullPenyebab: penyebab,
        count,
        fill: PIE_COLORS[index % PIE_COLORS.length],
        percentage: responses.length > 0 ? ((count / responses.length) * 100).toFixed(1) : '0',
      }))
      .sort((a, b) => b.count - a.count);

    // Teknologi Lain
    const teknologiLainGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      if (response.teknologi_lain && response.teknologi_lain.trim() !== '' && response.teknologi_lain !== 'Tidak Ada') {
        const teknologi = response.teknologi_lain.trim();
        teknologiLainGroups[teknologi] = (teknologiLainGroups[teknologi] || 0) + 1;
      }
    });

    const teknologiLainData = Object.entries(teknologiLainGroups)
      .map(([teknologi, count], index) => ({
        teknologi: teknologi.length > 20 ? teknologi.slice(0, 20) + '...' : teknologi,
        fullTeknologi: teknologi,
        count,
        fill: PIE_COLORS[index % PIE_COLORS.length],
        percentage: responses.length > 0 ? ((count / responses.length) * 100).toFixed(1) : '0',
      }))
      .sort((a, b) => b.count - a.count);

    // Overall Adoption
    const adoptionLevels = { Tinggi: 0, Sedang: 0, Rendah: 0 };
    responses.forEach((response: any) => {
      const scores = [
        (response.pembajakan_lahan_modern === 'Ya' || response.pembajakan_lahan_modern === 'ya') ? 1 : 0,
        (response.pengairan_sumur_bor === 'Ya' || response.pengairan_sumur_bor === 'ya') ? 1 : 0,
        (response.pengairan_pompa_air === 'Ya' || response.pengairan_pompa_air === 'ya') ? 1 : 0,
        (response.penyemprotan_pompa_elektrik === 'Ya' || response.penyemprotan_pompa_elektrik === 'ya') ? 1 : 0,
        (response.panen_mesin_potong === 'Ya' || response.panen_mesin_potong === 'ya') ? 1 : 0,
        (response.pakai_internet === 'Ya' || response.pakai_internet === 'ya') ? 1 : 0,
      ];
      const totalScore = scores.reduce((sum, score) => sum + score, 0);
      if (totalScore >= 5) adoptionLevels.Tinggi++;
      else if (totalScore >= 3) adoptionLevels.Sedang++;
      else adoptionLevels.Rendah++;
    });

    const overallAdoptionData = [
      { level: "Tinggi", count: adoptionLevels.Tinggi, fill: "hsl(142, 76%, 36%)" },
      { level: "Sedang", count: adoptionLevels.Sedang, fill: "hsl(38, 92%, 50%)" },
      { level: "Rendah", count: adoptionLevels.Rendah, fill: "hsl(0, 84%, 60%)" },
    ];
    const maxAdoption = Math.max(...Object.values(adoptionLevels));

    // Regional
    const regionalGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      const region = selectedKabupaten === "all" ? response.kab_kota : response.kecamatan;
      if (region && region.trim() !== '') {
        regionalGroups[region.trim()] = (regionalGroups[region.trim()] || 0) + 1;
      }
    });

    const regionalData = Object.entries(regionalGroups)
      .map(([region, count], index) => ({
        region: region.length > 15 ? region.slice(0, 15) + '...' : region,
        fullRegion: region,
        count,
        fill: PIE_COLORS[index % PIE_COLORS.length],
        percentage: responses.length > 0 ? ((count / responses.length) * 100).toFixed(1) : '0',
      }))
      .sort((a, b) => b.count - a.count);

    // Statistics
    const internetUsers = responses.filter((r: any) => r.pakai_internet === 'Ya' || r.pakai_internet === 'ya').length;
    const farmerGroupMembers = membershipYa;
    const modernizationRate = responses.length > 0 
      ? (technologyData.reduce((sum, item) => sum + item.ya, 0) / (technologyData.length * responses.length) * 100) 
      : 0;

    return {
      technology: technologyData,
      irrigation: irrigationData,
      spraying: sprayingData,
      harvest: harvestData,
      totalHarvest,
      information: informationData,
      membership: membershipData,
      totalMembership,
      climateAwareness: climateAwarenessData,
      failureAnalysis,
      totalFailure,
      penyebabGagal: penyebabGagalData,
      teknologiLain: teknologiLainData,
      responPergeseran: responPergeseranData,
      overallAdoption: overallAdoptionData,
      maxAdoption,
      regional: regionalData,
      totalResponses: responses.length,
      modernizationRate: modernizationRate.toFixed(1),
      internetUsage: internetUsers,
      farmerGroupMembership: farmerGroupMembers,
    };
  }, [filteredData, selectedKabupaten]);

  const renderChart = () => {
    if (!processedData) return null;

    switch (selectedChart) {
      case "technology":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Adopsi Teknologi Modern</CardTitle>
              <CardDescription className="text-xs">Perbandingan penggunaan teknologi</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[280px] w-full">
                <BarChart data={processedData.technology} layout="vertical" margin={{ left: 0, right: 40 }}>
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <YAxis 
                    dataKey="category" 
                    type="category"
                    tickLine={false} 
                    axisLine={false}
                    width={100}
                    tick={{ fontSize: 11 }}
                  />
                  <XAxis type="number" hide />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="ya" name="Ya" stackId="a" fill="hsl(142, 76%, 36%)" radius={[0, 0, 0, 0]} />
                  <Bar dataKey="tidak" name="Tidak" stackId="a" fill="hsl(0, 84%, 60%)" radius={[0, 4, 4, 0]}>
                    <LabelList 
                      dataKey="ya" 
                      position="insideLeft" 
                      className="fill-white text-[10px] font-medium"
                      formatter={(value: number) => value > 0 ? value : ''}
                    />
                  </Bar>
                </BarChart>
              </ChartContainer>
              <div className="flex justify-center gap-6 mt-3">
                <div className="flex items-center gap-2 text-xs">
                  <div className="h-3 w-3 rounded-sm bg-green-600" />
                  <span className="text-muted-foreground">Ya (Menggunakan)</span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <div className="h-3 w-3 rounded-sm bg-red-500" />
                  <span className="text-muted-foreground">Tidak</span>
                </div>
              </div>
            </CardContent>
          </Card>
        );

      case "irrigation":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Sistem Pengairan</CardTitle>
              <CardDescription className="text-xs">Metode pengairan yang digunakan</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[250px] w-full">
                <BarChart data={processedData.irrigation} margin={{ top: 20 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <XAxis dataKey="method" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                  <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={35} />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {processedData.irrigation.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="count" position="top" className="fill-foreground text-xs font-medium" />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "spraying":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Teknologi Penyemprotan</CardTitle>
              <CardDescription className="text-xs">Jenis alat penyemprotan</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[250px] w-full">
                <BarChart data={processedData.spraying} margin={{ top: 20 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <XAxis dataKey="type" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                  <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={35} />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {processedData.spraying.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="count" position="top" className="fill-foreground text-xs font-medium" />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "harvest":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Metode Panen</CardTitle>
              <CardDescription className="text-xs">Perbandingan metode panen</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="mx-auto h-[220px] w-full">
                <PieChart>
                  <ChartTooltip content={<ChartTooltipContent nameKey="method" />} />
                  <Pie
                    data={processedData.harvest}
                    dataKey="count"
                    nameKey="method"
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    strokeWidth={3}
                    stroke="hsl(var(--background))"
                  >
                    {processedData.harvest.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <Label
                      content={({ viewBox }) => {
                        if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                          return (
                            <text x={viewBox.cx} y={viewBox.cy} textAnchor="middle" dominantBaseline="middle">
                              <tspan x={viewBox.cx} y={viewBox.cy} className="fill-foreground text-2xl font-bold">
                                {processedData.totalHarvest}
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
                {processedData.harvest.map((item) => (
                  <div key={item.method} className="flex items-center gap-2 text-sm">
                    <div className="h-3 w-3 rounded-full" style={{ backgroundColor: item.fill }} />
                    <span className="text-muted-foreground">{item.method}</span>
                    <span className="font-semibold">{item.count}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        );

      case "information":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Sumber Informasi</CardTitle>
              <CardDescription className="text-xs">Sumber informasi pertanian</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[250px] w-full">
                <BarChart data={processedData.information} margin={{ top: 20 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <XAxis dataKey="source" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                  <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={35} />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {processedData.information.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="count" position="top" className="fill-foreground text-xs font-medium" />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "membership":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Keanggotaan Kelompok Tani</CardTitle>
              <CardDescription className="text-xs">Status keanggotaan organisasi</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="mx-auto h-[220px] w-full">
                <PieChart>
                  <ChartTooltip content={<ChartTooltipContent nameKey="status" />} />
                  <Pie
                    data={processedData.membership}
                    dataKey="count"
                    nameKey="status"
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    strokeWidth={3}
                    stroke="hsl(var(--background))"
                  >
                    {processedData.membership.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <Label
                      content={({ viewBox }) => {
                        if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                          return (
                            <text x={viewBox.cx} y={viewBox.cy} textAnchor="middle" dominantBaseline="middle">
                              <tspan x={viewBox.cx} y={viewBox.cy} className="fill-foreground text-2xl font-bold">
                                {processedData.totalMembership}
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
                {processedData.membership.map((item) => (
                  <div key={item.status} className="flex items-center gap-2 text-sm">
                    <div className="h-3 w-3 rounded-full" style={{ backgroundColor: item.fill }} />
                    <span className="text-muted-foreground">{item.status}</span>
                    <span className="font-semibold">{item.count}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        );

      case "climate-awareness":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Kesadaran Iklim</CardTitle>
              <CardDescription className="text-xs">Pengetahuan tentang perubahan iklim</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[250px] w-full">
                <BarChart data={processedData.climateAwareness} margin={{ top: 20 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <XAxis dataKey="awareness" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                  <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={35} />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {processedData.climateAwareness.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="count" position="top" className="fill-foreground text-xs font-medium" />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "failure-analysis":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Analisis Kegagalan Panen</CardTitle>
              <CardDescription className="text-xs">Riwayat kegagalan panen</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="mx-auto h-[220px] w-full">
                <PieChart>
                  <ChartTooltip content={<ChartTooltipContent nameKey="status" />} />
                  <Pie
                    data={processedData.failureAnalysis}
                    dataKey="count"
                    nameKey="status"
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    strokeWidth={3}
                    stroke="hsl(var(--background))"
                  >
                    {processedData.failureAnalysis.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <Label
                      content={({ viewBox }) => {
                        if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                          return (
                            <text x={viewBox.cx} y={viewBox.cy} textAnchor="middle" dominantBaseline="middle">
                              <tspan x={viewBox.cx} y={viewBox.cy} className="fill-foreground text-2xl font-bold">
                                {processedData.totalFailure}
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
                {processedData.failureAnalysis.map((item) => (
                  <div key={item.status} className="flex items-center gap-2 text-sm">
                    <div className="h-3 w-3 rounded-full" style={{ backgroundColor: item.fill }} />
                    <span className="text-muted-foreground">{item.status}</span>
                    <span className="font-semibold">{item.count}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        );

      case "penyebab-gagal":
        if (!processedData.penyebabGagal || processedData.penyebabGagal.length === 0) {
          return (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>Tidak ada data penyebab kegagalan</AlertDescription>
            </Alert>
          );
        }
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Penyebab Kegagalan</CardTitle>
              <CardDescription className="text-xs">Faktor penyebab gagal panen</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[300px] w-full">
                <BarChart data={processedData.penyebabGagal} layout="vertical" margin={{ left: 0, right: 40 }}>
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <YAxis 
                    dataKey="penyebab" 
                    type="category"
                    tickLine={false} 
                    axisLine={false}
                    width={120}
                    tick={{ fontSize: 10 }}
                  />
                  <XAxis type="number" hide />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {processedData.penyebabGagal.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="count" position="right" className="fill-foreground text-xs font-medium" />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "teknologi-lain":
        if (!processedData.teknologiLain || processedData.teknologiLain.length === 0) {
          return (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>Tidak ada data teknologi tambahan</AlertDescription>
            </Alert>
          );
        }
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Teknologi Tambahan</CardTitle>
              <CardDescription className="text-xs">Teknologi lain yang digunakan</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[300px] w-full">
                <BarChart data={processedData.teknologiLain} layout="vertical" margin={{ left: 0, right: 40 }}>
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <YAxis 
                    dataKey="teknologi" 
                    type="category"
                    tickLine={false} 
                    axisLine={false}
                    width={120}
                    tick={{ fontSize: 10 }}
                  />
                  <XAxis type="number" hide />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {processedData.teknologiLain.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="count" position="right" className="fill-foreground text-xs font-medium" />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "respon-pergeseran":
        if (!processedData.responPergeseran || processedData.responPergeseran.length === 0) {
          return (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>Tidak ada data respon pergeseran</AlertDescription>
            </Alert>
          );
        }
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Respon Pergeseran Musim</CardTitle>
              <CardDescription className="text-xs">Cara petani merespons perubahan musim</CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer 
                config={chartConfig} 
                className="w-full"
                style={{ height: Math.max(250, processedData.responPergeseran.length * 35) }}
              >
                <BarChart data={processedData.responPergeseran} layout="vertical" margin={{ left: 0, right: 40 }}>
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <YAxis 
                    dataKey="respon" 
                    type="category"
                    tickLine={false} 
                    axisLine={false}
                    width={150}
                    tick={{ fontSize: 10 }}
                  />
                  <XAxis type="number" hide />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {processedData.responPergeseran.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                    <LabelList dataKey="count" position="right" className="fill-foreground text-xs font-medium" />
                  </Bar>
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        );

      case "overall-adoption":
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Tingkat Adopsi Teknologi</CardTitle>
              <CardDescription className="text-xs">Kategori adopsi berdasarkan skor</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <ChartContainer config={chartConfig} className="mx-auto h-[220px] w-full">
                  <RadialBarChart
                    data={processedData.overallAdoption}
                    innerRadius={30}
                    outerRadius={100}
                    startAngle={180}
                    endAngle={0}
                  >
                    <ChartTooltip cursor={false} content={<ChartTooltipContent nameKey="level" />} />
                    <PolarAngleAxis type="number" domain={[0, processedData.maxAdoption]} tick={false} />
                    <RadialBar
                      dataKey="count"
                      background={{ fill: "hsl(var(--muted))" }}
                      cornerRadius={4}
                    >
                      {processedData.overallAdoption.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </RadialBar>
                  </RadialBarChart>
                </ChartContainer>
                <div className="flex flex-col justify-center gap-3">
                  {processedData.overallAdoption.map((item) => (
                    <div key={item.level} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: item.fill }} />
                        <span className="text-sm text-muted-foreground">{item.level}</span>
                      </div>
                      <span className="text-sm font-semibold tabular-nums">{item.count}</span>
                    </div>
                  ))}
                  <div className="pt-2 border-t text-xs text-muted-foreground">
                    <p>Tinggi: 5-6 teknologi</p>
                    <p>Sedang: 3-4 teknologi</p>
                    <p>Rendah: 0-2 teknologi</p>
                  </div>
                </div>
              </div>
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
                <BarChart data={processedData.regional} layout="vertical" margin={{ left: 0, right: 40 }}>
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted/30" />
                  <YAxis 
                    dataKey="region" 
                    type="category"
                    tickLine={false} 
                    axisLine={false}
                    width={100}
                    tick={{ fontSize: 10 }}
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
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
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
                <Tractor className="h-4 w-4 text-blue-600" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Modernisasi</p>
                <p className="text-lg font-semibold tabular-nums">{processedData.modernizationRate}%</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-md bg-amber-500/10 p-2.5">
                <Sprout className="h-4 w-4 text-amber-600" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Kelompok Tani</p>
                <p className="text-lg font-semibold tabular-nums">{processedData.farmerGroupMembership}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="rounded-md bg-purple-500/10 p-2.5">
                <Wifi className="h-4 w-4 text-purple-600" />
              </div>
              <div>
                <p className="text-[11px] text-muted-foreground uppercase tracking-wide">Pengguna Internet</p>
                <p className="text-lg font-semibold tabular-nums">{processedData.internetUsage}</p>
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

            <div className="min-w-[200px] space-y-1">
              <label className="text-[11px] text-muted-foreground uppercase tracking-wide flex items-center gap-1">
                <BarChart3 className="h-3 w-3" />
                Jenis Analisis
              </label>
              <Select value={selectedChart} onValueChange={(v: ManagementChartType) => setSelectedChart(v)}>
                <SelectTrigger className="h-8 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="technology">Teknologi Modern</SelectItem>
                  <SelectItem value="irrigation">Sistem Pengairan</SelectItem>
                  <SelectItem value="spraying">Penyemprotan</SelectItem>
                  <SelectItem value="harvest">Metode Panen</SelectItem>
                  <SelectItem value="information">Sumber Informasi</SelectItem>
                  <SelectItem value="membership">Keanggotaan</SelectItem>
                  <SelectItem value="climate-awareness">Kesadaran Iklim</SelectItem>
                  <SelectItem value="failure-analysis">Analisis Kegagalan</SelectItem>
                  <SelectItem value="penyebab-gagal">Penyebab Gagal</SelectItem>
                  <SelectItem value="teknologi-lain">Teknologi Lain</SelectItem>
                  <SelectItem value="respon-pergeseran">Respon Pergeseran</SelectItem>
                  <SelectItem value="overall-adoption">Adopsi Keseluruhan</SelectItem>
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