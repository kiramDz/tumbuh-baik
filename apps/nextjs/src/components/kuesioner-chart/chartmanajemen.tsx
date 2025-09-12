"use client";

import { useQuery } from "@tanstack/react-query";
import { getAllKuesionerManajemen } from "@/lib/fetch/files.fetch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, AreaChart, Area } from "recharts";
import { useState, useMemo } from "react";
import { BarChart3, PieChart as PieChartIcon, TrendingUp, Users, MapPin, Building2, Wifi, Tractor, Droplets, Sprout, Scissors, Info, UserCheck, AlertTriangle, Settings } from "lucide-react";

const COLORS = ["#16a34a", "#0ea5e9", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#84cc16", "#f97316"];

type ManagementChartType = "technology" | "irrigation" | "spraying" | "harvest" | "information" | "membership" | "climate-awareness" | "failure-analysis" | "overall-adoption" | "regional";

export default function ChartManajemen() {
  const [selectedChart, setSelectedChart] = useState<ManagementChartType>("technology");
  const [selectedKabupaten, setSelectedKabupaten] = useState<string>("all");

  const {
    data: manajemenData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["manajemen-data"],
    queryFn: getAllKuesionerManajemen,
  });

  // Get unique kabupaten list
  const kabupatenList = useMemo(() => {
    if (!manajemenData) return [];
    
    const uniqueKabupaten = [...new Set(manajemenData.map((item: any) => item.kab_kota))];
    return uniqueKabupaten.filter(Boolean).sort();
  }, [manajemenData]);

  // Filter data berdasarkan kabupaten yang dipilih
  const filteredData = useMemo(() => {
    if (!manajemenData) return [];
    
    if (selectedKabupaten === "all") {
      return manajemenData;
    }
    
    return manajemenData.filter((item: any) => item.kab_kota === selectedKabupaten);
  }, [manajemenData, selectedKabupaten]);

  // Process data untuk charts berdasarkan schema yang sebenarnya
  const processedData = useMemo(() => {
    if (!filteredData || filteredData.length === 0) return null;
    
    const responses = filteredData;
    
    // Technology Adoption - Modern farming tools
    const technologyData = [
      {
        category: "Pembajakan Lahan Modern",
        ya: responses.filter((r: any) => r.pembajakan_lahan_modern?.toLowerCase() === 'ya').length,
        tidak: responses.filter((r: any) => r.pembajakan_lahan_modern?.toLowerCase() === 'tidak').length,
      },
      {
        category: "Panen Mesin Potong",
        ya: responses.filter((r: any) => r.panen_mesin_potong?.toLowerCase() === 'ya').length,
        tidak: responses.filter((r: any) => r.panen_mesin_potong?.toLowerCase() === 'tidak').length,
      },
      {
        category: "Pakai Internet",
        ya: responses.filter((r: any) => r.pakai_internet?.toLowerCase() === 'ya').length,
        tidak: responses.filter((r: any) => r.pakai_internet?.toLowerCase() === 'tidak').length,
      },
    ];

    // Irrigation Technology
    const irrigationData = [
      {
        method: "Sumur Bor",
        ya: responses.filter((r: any) => r.pengairan_sumur_bor?.toLowerCase() === 'ya').length,
        tidak: responses.filter((r: any) => r.pengairan_sumur_bor?.toLowerCase() === 'tidak').length,
        percentage: ((responses.filter((r: any) => r.pengairan_sumur_bor?.toLowerCase() === 'ya').length / responses.length) * 100).toFixed(1),
      },
      {
        method: "Pompa Air",
        ya: responses.filter((r: any) => r.pengairan_pompa_air?.toLowerCase() === 'ya').length,
        tidak: responses.filter((r: any) => r.pengairan_pompa_air?.toLowerCase() === 'tidak').length,
        percentage: ((responses.filter((r: any) => r.pengairan_pompa_air?.toLowerCase() === 'ya').length / responses.length) * 100).toFixed(1),
      },
    ];

    // Spraying Technology
    const sprayingData = [
      {
        type: "Pompa Tangan",
        ya: responses.filter((r: any) => r.penyemprotan_pompa_tangan?.toLowerCase() === 'ya').length,
        tidak: responses.filter((r: any) => r.penyemprotan_pompa_tangan?.toLowerCase() === 'tidak').length,
        percentage: ((responses.filter((r: any) => r.penyemprotan_pompa_tangan?.toLowerCase() === 'ya').length / responses.length) * 100).toFixed(1),
      },
      {
        type: "Pompa Elektrik",
        ya: responses.filter((r: any) => r.penyemprotan_pompa_elektrik?.toLowerCase() === 'ya').length,
        tidak: responses.filter((r: any) => r.penyemprotan_pompa_elektrik?.toLowerCase() === 'tidak').length,
        percentage: ((responses.filter((r: any) => r.penyemprotan_pompa_elektrik?.toLowerCase() === 'ya').length / responses.length) * 100).toFixed(1),
      },
    ];

    // Harvest Technology
    const harvestData = [
      {
        method: "Mesin Potong",
        count: responses.filter((r: any) => r.panen_mesin_potong?.toLowerCase() === 'ya').length,
        percentage: ((responses.filter((r: any) => r.panen_mesin_potong?.toLowerCase() === 'ya').length / responses.length) * 100).toFixed(1),
      },
      {
        method: "Manual/Tradisional",
        count: responses.filter((r: any) => r.panen_mesin_potong?.toLowerCase() === 'tidak').length,
        percentage: ((responses.filter((r: any) => r.panen_mesin_potong?.toLowerCase() === 'tidak').length / responses.length) * 100).toFixed(1),
      },
    ];

    // Information Sources - Debug version
    const informationData = [
      {
        source: "Penyuluh",
        ya: responses.filter((r: any) => {
          console.log('Penyuluh data:', r.info_penyuluh); // Debug log
          return r.info_penyuluh?.toLowerCase() === 'ya';
        }).length,
        tidak: responses.filter((r: any) => r.info_penyuluh?.toLowerCase() === 'tidak').length,
        percentage: ((responses.filter((r: any) => r.info_penyuluh?.toLowerCase() === 'ya').length / responses.length) * 100).toFixed(1),
      },
      {
        source: "Keuchik",
        ya: responses.filter((r: any) => {
          console.log('Keuchik data:', r.info_keuchik); // Debug log
          return r.info_keuchik?.toLowerCase() === 'ya';
        }).length,
        tidak: responses.filter((r: any) => r.info_keuchik?.toLowerCase() === 'tidak').length,
        percentage: ((responses.filter((r: any) => r.info_keuchik?.toLowerCase() === 'ya').length / responses.length) * 100).toFixed(1),
      },
      {
        source: "Keujrun Blang",
        ya: responses.filter((r: any) => {
          console.log('Keujrun Blang data:', r.info_keujrun_blang); // Debug log
          return r.info_keujrun_blang?.toLowerCase() === 'ya';
        }).length,
        tidak: responses.filter((r: any) => r.info_keujrun_blang?.toLowerCase() === 'tidak').length,
        percentage: ((responses.filter((r: any) => r.info_keujrun_blang?.toLowerCase() === 'ya').length / responses.length) * 100).toFixed(1),
      },
    ];

    // Log untuk debugging
    console.log('Information Data:', informationData);
    console.log('Sample response:', responses[0]);

    // Membership & Organization
    const membershipData = [
      {
        organization: "Kelompok Tani",
        ya: responses.filter((r: any) => r.anggota_kelompok_tani?.toLowerCase() === 'ya').length,
        tidak: responses.filter((r: any) => r.anggota_kelompok_tani?.toLowerCase() === 'tidak').length,
        percentage: ((responses.filter((r: any) => r.anggota_kelompok_tani?.toLowerCase() === 'ya').length / responses.length) * 100).toFixed(1),
      },
    ];

    // Climate Awareness
    const climateAwarenessData = [
      {
        awareness: "Tahu KATAM",
        ya: responses.filter((r: any) => r.tahu_katam?.toLowerCase() === 'ya').length,
        tidak: responses.filter((r: any) => r.tahu_katam?.toLowerCase() === 'tidak').length,
      },
      {
        awareness: "Tahu Pergeseran Musim",
        ya: responses.filter((r: any) => r.tahu_pergeseran_musim?.toLowerCase() === 'ya').length,
        tidak: responses.filter((r: any) => r.tahu_pergeseran_musim?.toLowerCase() === 'tidak').length,
      },
    ];

    // Failure Analysis
    const failureData = responses.reduce((acc: any, response: any) => {
      const gagal = response.pernah_gagal_tanam?.toLowerCase() === 'ya' ? 'Ya' : 'Tidak';
      acc[gagal] = (acc[gagal] || 0) + 1;
      return acc;
    }, {});

    const failureAnalysis = Object.entries(failureData).map(([status, count]) => ({
      status,
      count: count as number,
      percentage: (((count as number) / responses.length) * 100).toFixed(1),
    }));

    // Penyebab Gagal analysis
    const penyebabGagal = responses
      .filter((r: any) => r.pernah_gagal_tanam?.toLowerCase() === 'ya' && r.penyebab_gagal)
      .reduce((acc: any, response: any) => {
        const penyebab = response.penyebab_gagal.trim();
        acc[penyebab] = (acc[penyebab] || 0) + 1;
        return acc;
      }, {});

    const penyebabGagalData = Object.entries(penyebabGagal).map(([penyebab, count]) => ({
      penyebab,
      count: count as number,
    }));

    // Overall Technology Adoption Score
    const overallAdoptionData = responses.map((response: any, index: number) => {
      const scores = [
        response.pembajakan_lahan_modern?.toLowerCase() === 'ya' ? 1 : 0,
        response.pengairan_sumur_bor?.toLowerCase() === 'ya' ? 1 : 0,
        response.pengairan_pompa_air?.toLowerCase() === 'ya' ? 1 : 0,
        response.penyemprotan_pompa_elektrik?.toLowerCase() === 'ya' ? 1 : 0,
        response.panen_mesin_potong?.toLowerCase() === 'ya' ? 1 : 0,
        response.pakai_internet?.toLowerCase() === 'ya' ? 1 : 0,
      ];
      const totalScore = scores.reduce((sum, score) => sum + score, 0);
      return {
        petani: `Petani ${index + 1}`,
        score: totalScore,
        level: totalScore >= 5 ? 'Tinggi' : totalScore >= 3 ? 'Sedang' : 'Rendah'
      };
    });

    const adoptionLevels = overallAdoptionData.reduce((acc: any, item: { petani: string; score: number; level: string }) => {
      acc[item.level] = (acc[item.level] || 0) + 1;
      return acc;
    }, {});

    const adoptionSummary = Object.entries(adoptionLevels).map(([level, count]) => ({
      level,
      count: count as number,
      percentage: (((count as number) / responses.length) * 100).toFixed(1),
    }));

    // Regional distribution (Kecamatan level when kabupaten is selected)
    const regionalGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      const region = selectedKabupaten === "all" ? response.kab_kota : response.kecamatan;
      if (region && region.trim() !== '') {
        regionalGroups[region.trim()] = (regionalGroups[region.trim()] || 0) + 1;
      }
    });

    const regional = Object.entries(regionalGroups).map(([region, count]) => ({
      region,
      count,
      percentage: ((count / responses.length) * 100).toFixed(1),
    }));

    return {
      technology: technologyData,
      irrigation: irrigationData,
      spraying: sprayingData,
      harvest: harvestData,
      information: informationData,
      membership: membershipData,
      climateAwareness: climateAwarenessData,
      failureAnalysis,
      penyebabGagal: penyebabGagalData,
      overallAdoption: adoptionSummary,
      regional,
      totalResponses: responses.length,
      modernizationRate: ((technologyData.reduce((sum, item) => sum + item.ya, 0) / (technologyData.length * responses.length)) * 100).toFixed(1),
    };
  }, [filteredData, selectedKabupaten]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-md">
          <p className="font-medium text-gray-900">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const renderTechnologyChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={processedData?.technology}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis 
          dataKey="category" 
          angle={-45} 
          textAnchor="end" 
          height={100}
          tick={{ fontSize: 12, fill: '#64748b' }}
        />
        <YAxis tick={{ fontSize: 12, fill: '#64748b' }} />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="ya" fill="#16a34a" radius={[4, 4, 0, 0]} name="Ya" />
        <Bar dataKey="tidak" fill="#ef4444" radius={[4, 4, 0, 0]} name="Tidak" />
      </BarChart>
    </ResponsiveContainer>
  );

  const renderIrrigationChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <PieChart>
        <Pie
          data={processedData?.irrigation}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ method, percentage }) => `${method}: ${percentage}%`}
          outerRadius={120}
          fill="#8884d8"
          dataKey="ya"
        >
          {processedData?.irrigation.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
      </PieChart>
    </ResponsiveContainer>
  );

  const renderSprayingChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={processedData?.spraying}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="type" tick={{ fontSize: 12, fill: '#64748b' }} />
        <YAxis tick={{ fontSize: 12, fill: '#64748b' }} />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="ya" fill="#0ea5e9" radius={[4, 4, 0, 0]} name="Ya" />
        <Bar dataKey="tidak" fill="#ef4444" radius={[4, 4, 0, 0]} name="Tidak" />
      </BarChart>
    </ResponsiveContainer>
  );

  const renderHarvestChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <PieChart>
        <Pie
          data={processedData?.harvest}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ method, percentage }) => `${method}: ${percentage}%`}
          outerRadius={120}
          fill="#8884d8"
          dataKey="count"
        >
          {processedData?.harvest.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
      </PieChart>
    </ResponsiveContainer>
  );

  const renderInformationChart = () => {
    // Debug: Log data yang akan dirender
    console.log('Information chart data:', processedData?.information);
    
    // Check if data exists
    if (!processedData?.information || processedData.information.length === 0) {
      return (
        <div className="flex items-center justify-center h-96 text-gray-500">
          <div className="text-center">
            <Info className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="font-medium">Tidak ada data sumber informasi</p>
            <p className="text-sm">Data kosong atau belum tersedia</p>
          </div>
        </div>
      );
    }

    // Check if all values are zero
    const totalData = processedData.information.reduce((sum, item) => sum + item.ya + item.tidak, 0);
    if (totalData === 0) {
      return (
        <div className="flex items-center justify-center h-96 text-gray-500">
          <div className="text-center">
            <Info className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="font-medium">Semua data sumber informasi bernilai 0</p>
            <p className="text-sm">Periksa data di database atau field mapping</p>
            <div className="mt-4 p-3 bg-gray-100 rounded text-xs text-left">
              <p className="font-semibold mb-2">Debug info:</p>
              {processedData.information.map((item, index) => (
                <p key={index}>{item.source}: Ya={item.ya}, Tidak={item.tidak}</p>
              ))}
            </div>
          </div>
        </div>
      );
    }

    // Render Vertical Bar Chart
    return (
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={processedData.information}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis 
            dataKey="source" 
            tick={{ fontSize: 12, fill: '#64748b' }}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis tick={{ fontSize: 12, fill: '#64748b' }} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="ya" fill="#16a34a" radius={[4, 4, 0, 0]} name="Ya" />
          <Bar dataKey="tidak" fill="#ef4444" radius={[4, 4, 0, 0]} name="Tidak" />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const renderMembershipChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <PieChart>
        <Pie
          data={[
            { name: "Anggota", value: processedData?.membership[0]?.ya || 0 },
            { name: "Bukan Anggota", value: processedData?.membership[0]?.tidak || 0 }
          ]}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ name, value }) => `${name}: ${value}`}
          outerRadius={120}
          fill="#8884d8"
          dataKey="value"
        >
          <Cell fill="#16a34a" />
          <Cell fill="#ef4444" />
        </Pie>
        <Tooltip content={<CustomTooltip />} />
      </PieChart>
    </ResponsiveContainer>
  );

  const renderClimateAwarenessChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={processedData?.climateAwareness}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="awareness" tick={{ fontSize: 12, fill: '#64748b' }} />
        <YAxis tick={{ fontSize: 12, fill: '#64748b' }} />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="ya" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Ya" />
        <Bar dataKey="tidak" fill="#ef4444" radius={[4, 4, 0, 0]} name="Tidak" />
      </BarChart>
    </ResponsiveContainer>
  );

  const renderFailureAnalysisChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <div className="space-y-4">
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={processedData?.failureAnalysis}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ status, percentage }) => `${status}: ${percentage}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="count"
              >
                {processedData?.failureAnalysis.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.status === 'Ya' ? "#ef4444" : "#16a34a"} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        {processedData?.penyebabGagal && processedData.penyebabGagal.length > 0 && (
          <div className="h-48">
            <p className="text-sm font-medium mb-2">Penyebab Kegagalan:</p>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={processedData.penyebabGagal} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis type="number" tick={{ fontSize: 10, fill: '#64748b' }} />
                <YAxis dataKey="penyebab" type="category" width={120} tick={{ fontSize: 10, fill: '#64748b' }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" fill="#f97316" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </ResponsiveContainer>
  );

  const renderOverallAdoptionChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <AreaChart data={processedData?.overallAdoption}>
        <defs>
          <linearGradient id="adoptionGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#16a34a" stopOpacity={0.8}/>
            <stop offset="95%" stopColor="#16a34a" stopOpacity={0.1}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="level" tick={{ fontSize: 12, fill: '#64748b' }} />
        <YAxis tick={{ fontSize: 12, fill: '#64748b' }} />
        <Tooltip content={<CustomTooltip />} />
        <Area type="monotone" dataKey="count" stroke="#16a34a" fillOpacity={1} fill="url(#adoptionGradient)" />
      </AreaChart>
    </ResponsiveContainer>
  );

  const renderRegionalChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={processedData?.regional}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="region" angle={-45} textAnchor="end" height={100} tick={{ fontSize: 12, fill: '#64748b' }} />
        <YAxis tick={{ fontSize: 12, fill: '#64748b' }} />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );

  const getChartTitle = () => {
    const kabupatenText = selectedKabupaten === "all" ? "Semua Kabupaten" : selectedKabupaten;
    
    switch (selectedChart) {
      case "technology": return `Adopsi Teknologi Modern - ${kabupatenText}`;
      case "irrigation": return `Sistem Pengairan - ${kabupatenText}`;
      case "spraying": return `Teknologi Penyemprotan - ${kabupatenText}`;
      case "harvest": return `Metode Panen - ${kabupatenText}`;
      case "information": return `Sumber Informasi - ${kabupatenText}`;
      case "membership": return `Keanggotaan Organisasi - ${kabupatenText}`;
      case "climate-awareness": return `Kesadaran Iklim - ${kabupatenText}`;
      case "failure-analysis": return `Analisis Kegagalan Panen - ${kabupatenText}`;
      case "overall-adoption": return `Tingkat Adopsi Teknologi - ${kabupatenText}`;
      case "regional": return selectedKabupaten === "all" ? "Distribusi Per Kabupaten" : `Distribusi Per Kecamatan - ${kabupatenText}`;
      default: return "Chart Manajemen Usaha";
    }
  };

  const getChartDescription = () => {
    const totalData = manajemenData?.length || 0;
    const filteredCount = filteredData?.length || 0;
    const kabupatenText = selectedKabupaten === "all" ? "semua kabupaten" : selectedKabupaten;
    
    switch (selectedChart) {
      case "technology": return `Adopsi teknologi modern dalam usaha tani di ${kabupatenText} (${filteredCount} dari ${totalData} responden)`;
      case "irrigation": return `Penggunaan sistem pengairan modern di ${kabupatenText}`;
      case "spraying": return `Teknologi yang digunakan untuk penyemprotan di ${kabupatenText}`;
      case "harvest": return `Metode panen yang digunakan petani di ${kabupatenText}`;
      case "information": return `Sumber informasi pertanian yang diakses di ${kabupatenText}`;
      case "membership": return `Partisipasi dalam kelompok tani di ${kabupatenText}`;
      case "climate-awareness": return `Kesadaran terhadap perubahan iklim dan program KATAM di ${kabupatenText}`;
      case "failure-analysis": return `Analisis kegagalan panen dan penyebabnya di ${kabupatenText}`;
      case "overall-adoption": return `Tingkat adopsi teknologi secara keseluruhan di ${kabupatenText}`;
      case "regional": return selectedKabupaten === "all" ? "Distribusi manajemen berdasarkan kabupaten" : `Distribusi manajemen berdasarkan kecamatan di ${kabupatenText}`;
      default: return "Analisis data manajemen usaha";
    }
  };

  const getChartIcon = () => {
    switch (selectedChart) {
      case "technology": return <Tractor className="w-5 h-5" />;
      case "irrigation": return <Droplets className="w-5 h-5" />;
      case "spraying": return <Sprout className="w-5 h-5" />;
      case "harvest": return <Scissors className="w-5 h-5" />;
      case "information": return <Info className="w-5 h-5" />;
      case "membership": return <UserCheck className="w-5 h-5" />;
      case "climate-awareness": return <AlertTriangle className="w-5 h-5" />;
      case "failure-analysis": return <BarChart3 className="w-5 h-5" />;
      case "overall-adoption": return <Settings className="w-5 h-5" />;
      case "regional": return <MapPin className="w-5 h-5" />;
      default: return <Building2 className="w-5 h-5" />;
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-96 bg-gray-200 rounded"></div>
        </div>
        <p className="text-sm text-gray-500">Memuat data manajemen...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-red-600">
              <span className="font-medium">Error:</span>
              <span className="text-sm">{error instanceof Error ? error.message : "Unknown error"}</span>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!processedData) {
    return (
      <div className="space-y-4">
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-gray-500">
              <Building2 className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p className="font-medium">Tidak ada data untuk {selectedKabupaten === "all" ? "semua kabupaten" : selectedKabupaten}</p>
              <p className="text-sm">Silakan pilih kabupaten lain atau periksa data yang tersedia</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h2 className="text-lg font-semibold text-gray-900">Analisis Manajemen Usaha Tani</h2>
          <p className="text-sm text-gray-600">Teknologi, informasi, dan praktek manajemen usaha tani</p>
        </div>
        <Badge variant="secondary" className="px-3 py-1">
          <Building2 className="w-3 h-3 mr-1" />
          {processedData.totalResponses} Responden
        </Badge>
      </div>

      <Separator />

      {/* Filter Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Kabupaten Filter */}
        <Card className="bg-blue-50/50">
          <CardContent className="p-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-blue-700">
                <MapPin className="w-5 h-5" />
                <span className="text-sm font-medium">Filter Kabupaten:</span>
              </div>
              <Select value={selectedKabupaten} onValueChange={setSelectedKabupaten}>
                <SelectTrigger className="w-64 bg-white">
                  <SelectValue placeholder="Pilih kabupaten" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">🗺️ Semua Kabupaten</SelectItem>
                  {kabupatenList.map((kabupaten) => (
                    <SelectItem key={kabupaten as string} value={kabupaten as string}>
                      📍 {kabupaten as string}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Chart Type Selection */}
        <Card className="bg-gray-50/50">
          <CardContent className="p-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-gray-700">
                {getChartIcon()}
                <span className="text-sm font-medium">Jenis Analisis:</span>
              </div>
              <Select value={selectedChart} onValueChange={(value: ManagementChartType) => setSelectedChart(value)}>
                <SelectTrigger className="w-80 bg-white">
                  <SelectValue placeholder="Pilih jenis chart" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="technology">🚜 Teknologi Modern</SelectItem>
                  <SelectItem value="irrigation">💧 Sistem Pengairan</SelectItem>
                  <SelectItem value="spraying">🔧 Teknologi Penyemprotan</SelectItem>
                  <SelectItem value="harvest">✂️ Metode Panen</SelectItem>
                  <SelectItem value="information">ℹ️ Sumber Informasi</SelectItem>
                  <SelectItem value="membership">👥 Keanggotaan Organisasi</SelectItem>
                  <SelectItem value="climate-awareness">⚠️ Kesadaran Iklim</SelectItem>
                  <SelectItem value="failure-analysis">📊 Analisis Kegagalan</SelectItem>
                  <SelectItem value="overall-adoption">⚙️ Tingkat Adopsi</SelectItem>
                  <SelectItem value="regional">🗺️ Distribusi Wilayah</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Chart Display */}
      <Card className="shadow-sm">
        <CardHeader className="pb-4">
          <div className="flex items-center gap-2">
            {getChartIcon()}
            <div>
              <CardTitle className="text-lg">{getChartTitle()}</CardTitle>
              <CardDescription className="text-sm">{getChartDescription()}</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          {selectedChart === "technology" && renderTechnologyChart()}
          {selectedChart === "irrigation" && renderIrrigationChart()}
          {selectedChart === "spraying" && renderSprayingChart()}
          {selectedChart === "harvest" && renderHarvestChart()}
          {selectedChart === "information" && renderInformationChart()}
          {selectedChart === "membership" && renderMembershipChart()}
          {selectedChart === "climate-awareness" && renderClimateAwarenessChart()}
          {selectedChart === "failure-analysis" && renderFailureAnalysisChart()}
          {selectedChart === "overall-adoption" && renderOverallAdoptionChart()}
          {selectedChart === "regional" && renderRegionalChart()}
        </CardContent>
      </Card>

      {/* Statistics Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="hover:shadow-md transition-shadow duration-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Building2 className="w-4 h-4 text-green-600" />
              Total Responden
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{processedData.totalResponses}</div>
            <p className="text-xs text-muted-foreground">
              {selectedKabupaten === "all" ? "dari semua kabupaten" : `di ${selectedKabupaten}`}
            </p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow duration-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Tractor className="w-4 h-4 text-blue-600" />
              Modernisasi
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{processedData.modernizationRate}%</div>
            <p className="text-xs text-muted-foreground">tingkat modernisasi</p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow duration-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <UserCheck className="w-4 h-4 text-amber-600" />
              Kelompok Tani
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-amber-600">
              {processedData.membership[0]?.percentage || 0}%
            </div>
            <p className="text-xs text-muted-foreground">anggota kelompok tani</p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow duration-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Wifi className="w-4 h-4 text-purple-600" />
              Internet
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600">
              {processedData.technology.find(t => t.category === 'Pakai Internet')?.ya || 0}
            </div>
            <p className="text-xs text-muted-foreground">pengguna internet</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}