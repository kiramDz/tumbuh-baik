"use client";

import { useQuery } from "@tanstack/react-query";
import { getAllKuesionerPetani } from "@/lib/fetch/files.fetch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from "recharts";
import { useState, useMemo } from "react";
import { BarChart3, PieChart as PieChartIcon, TrendingUp, Users, MapPin, Filter } from "lucide-react";

const COLORS = ["#16a34a", "#0ea5e9", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];

type ChartType = "demographics" | "education" | "farming-experience" | "land-ownership" | "varieties" | "regional";

export default function ChartKuesioner() {
  const [selectedChart, setSelectedChart] = useState<ChartType>("demographics");
  const [selectedKabupaten, setSelectedKabupaten] = useState<string>("all");

  const {
    data: kuisionerData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["kuesioner-data"],
    queryFn: getAllKuesionerPetani,
  });

  // Get unique kabupaten list
  const kabupatenList = useMemo(() => {
    if (!kuisionerData) return [];
    
    const uniqueKabupaten = [...new Set(kuisionerData.map((item: any) => item.kab_kota))];
    return uniqueKabupaten.filter(Boolean).sort();
  }, [kuisionerData]);

  // Filter data berdasarkan kabupaten yang dipilih
  const filteredData = useMemo(() => {
    if (!kuisionerData) return [];
    
    if (selectedKabupaten === "all") {
      return kuisionerData;
    }
    
    return kuisionerData.filter((item: any) => item.kab_kota === selectedKabupaten);
  }, [kuisionerData, selectedKabupaten]);

  // Process data untuk charts
  const processedData = useMemo(() => {
    if (!filteredData || filteredData.length === 0) return null;
    
    const responses = filteredData;
    
    // Demographics - berdasarkan umur dan jenis kelamin
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

    const demographics = [
      ...Object.entries(ageGroups).map(([category, count]) => ({
        category: `Usia ${category}`,
        count,
        percentage: ((count / responses.length) * 100).toFixed(1),
        type: 'age'
      })),
      ...Object.entries(genderGroups).map(([category, count]) => ({
        category,
        count,
        percentage: ((count / responses.length) * 100).toFixed(1),
        type: 'gender'
      }))
    ];

    // Education level
    const educationGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      const education = response.pendidikan_terakhir;
      if (education && education.trim() !== '') {
        educationGroups[education.trim()] = (educationGroups[education.trim()] || 0) + 1;
      }
    });

    const education = Object.entries(educationGroups).map(([level, count]) => ({
      level,
      count,
      percentage: ((count / responses.length) * 100).toFixed(1),
    }));

    // Farming experience
    const currentYear = new Date().getFullYear();
    const experienceGroups = { "1-5 tahun": 0, "6-10 tahun": 0, "11-20 tahun": 0, "20+ tahun": 0 };

    responses.forEach((response: any) => {
      const startYear = response.tahun_mulai_bertani;
      if (startYear && startYear > 1900) {
        const experience = currentYear - startYear;
        if (experience >= 1 && experience <= 5) experienceGroups["1-5 tahun"]++;
        else if (experience >= 6 && experience <= 10) experienceGroups["6-10 tahun"]++;
        else if (experience >= 11 && experience <= 20) experienceGroups["11-20 tahun"]++;
        else if (experience > 20) experienceGroups["20+ tahun"]++;
      }
    });

    const farmingExperience = Object.entries(experienceGroups).map(([range, count]) => ({
      range,
      count,
      percentage: ((count / responses.length) * 100).toFixed(1),
    }));

    // Land ownership
    const landOwnershipGroups = { "< 1000 m²": 0, "1000-5000 m²": 0, "5000-10000 m²": 0, "> 10000 m²": 0 };

    responses.forEach((response: any) => {
      const totalLand = response.total_lahan_m2;
      if (totalLand && totalLand > 0) {
        if (totalLand < 1000) landOwnershipGroups["< 1000 m²"]++;
        else if (totalLand >= 1000 && totalLand < 5000) landOwnershipGroups["1000-5000 m²"]++;
        else if (totalLand >= 5000 && totalLand < 10000) landOwnershipGroups["5000-10000 m²"]++;
        else if (totalLand >= 10000) landOwnershipGroups["> 10000 m²"]++;
      }
    });

    const landOwnership = Object.entries(landOwnershipGroups).map(([range, count]) => ({
      range,
      count,
      percentage: ((count / responses.length) * 100).toFixed(1),
    }));

    // Rice varieties
    const varietiesGroups: { [key: string]: number } = {};
    responses.forEach((response: any) => {
      const variety = response.varietas_padi;
      if (variety && typeof variety === 'string' && variety.trim() !== '') {
        varietiesGroups[variety.trim()] = (varietiesGroups[variety.trim()] || 0) + 1;
      }
    });

    const varieties = Object.entries(varietiesGroups).map(([variety, count]) => ({
      variety,
      count,
      percentage: ((count / responses.length) * 100).toFixed(1),
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
      demographics,
      education,
      farmingExperience,
      landOwnership,
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

  const renderDemographicsChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={processedData?.demographics}>
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
        <Bar dataKey="count" fill="#16a34a" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );

  const renderEducationChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <PieChart>
        <Pie
          data={processedData?.education}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ level, percentage }) => `${level}: ${percentage}%`}
          outerRadius={120}
          fill="#8884d8"
          dataKey="count"
        >
          {processedData?.education.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
      </PieChart>
    </ResponsiveContainer>
  );

  const renderFarmingExperienceChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={processedData?.farmingExperience}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="range" tick={{ fontSize: 12, fill: '#64748b' }} />
        <YAxis tick={{ fontSize: 12, fill: '#64748b' }} />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="count" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );

  const renderLandOwnershipChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <PieChart>
        <Pie
          data={processedData?.landOwnership}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ range, percentage }) => `${range}: ${percentage}%`}
          outerRadius={120}
          fill="#8884d8"
          dataKey="count"
        >
          {processedData?.landOwnership.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
      </PieChart>
    </ResponsiveContainer>
  );

  const renderVarietiesChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={processedData?.varieties} layout="horizontal">
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis type="number" tick={{ fontSize: 12, fill: '#64748b' }} />
        <YAxis dataKey="variety" type="category" width={100} tick={{ fontSize: 12, fill: '#64748b' }} />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="count" fill="#f59e0b" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );

  const renderRegionalChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={processedData?.regional}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="region" angle={-45} textAnchor="end" height={100} tick={{ fontSize: 12, fill: '#64748b' }} />
        <YAxis tick={{ fontSize: 12, fill: '#64748b' }} />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );

  const getChartTitle = () => {
    const kabupatenText = selectedKabupaten === "all" ? "Semua Kabupaten" : selectedKabupaten;
    
    switch (selectedChart) {
      case "demographics": return `Demografi Petani - ${kabupatenText}`;
      case "education": return `Tingkat Pendidikan - ${kabupatenText}`;
      case "farming-experience": return `Pengalaman Bertani - ${kabupatenText}`;
      case "land-ownership": return `Kepemilikan Lahan - ${kabupatenText}`;
      case "varieties": return `Varietas Padi - ${kabupatenText}`;
      case "regional": return selectedKabupaten === "all" ? "Distribusi Per Kabupaten" : `Distribusi Per Kecamatan - ${kabupatenText}`;
      default: return "Chart Kuesioner Petani";
    }
  };

  const getChartDescription = () => {
    const totalData = kuisionerData?.length || 0;
    const filteredCount = filteredData?.length || 0;
    const kabupatenText = selectedKabupaten === "all" ? "semua kabupaten" : selectedKabupaten;
    
    switch (selectedChart) {
      case "demographics": return `Distribusi usia dan jenis kelamin petani di ${kabupatenText} (${filteredCount} dari ${totalData} petani)`;
      case "education": return `Tingkat pendidikan terakhir petani di ${kabupatenText}`;
      case "farming-experience": return `Pengalaman bertani berdasarkan tahun mulai di ${kabupatenText}`;
      case "land-ownership": return `Distribusi luas kepemilikan lahan di ${kabupatenText}`;
      case "varieties": return `Jenis varietas padi yang ditanam di ${kabupatenText}`;
      case "regional": return selectedKabupaten === "all" ? "Distribusi petani berdasarkan kabupaten" : `Distribusi petani berdasarkan kecamatan di ${kabupatenText}`;
      default: return "Analisis data kuesioner petani";
    }
  };

  const getChartIcon = () => {
    switch (selectedChart) {
      case "demographics": return <Users className="w-5 h-5" />;
      case "education": return <PieChartIcon className="w-5 h-5" />;
      case "farming-experience": return <TrendingUp className="w-5 h-5" />;
      case "land-ownership": return <PieChartIcon className="w-5 h-5" />;
      case "varieties": return <BarChart3 className="w-5 h-5" />;
      case "regional": return <MapPin className="w-5 h-5" />;
      default: return <BarChart3 className="w-5 h-5" />;
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-96 bg-gray-200 rounded"></div>
        </div>
        <p className="text-sm text-gray-500">Memuat data kuesioner...</p>
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
              <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
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
          <h2 className="text-lg font-semibold text-gray-900">Analisis Data Kuesioner</h2>
          <p className="text-sm text-gray-600">Visualisasi dan statistik data petani per kabupaten</p>
        </div>
        <Badge variant="secondary" className="px-3 py-1">
          <Users className="w-3 h-3 mr-1" />
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
              <Select value={selectedChart} onValueChange={(value: ChartType) => setSelectedChart(value)}>
                <SelectTrigger className="w-64 bg-white">
                  <SelectValue placeholder="Pilih jenis chart" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="demographics">📊 Demografi</SelectItem>
                  <SelectItem value="education">🎓 Pendidikan</SelectItem>
                  <SelectItem value="farming-experience">⏰ Pengalaman Bertani</SelectItem>
                  <SelectItem value="land-ownership">🏞️ Kepemilikan Lahan</SelectItem>
                  <SelectItem value="varieties">🌾 Varietas Padi</SelectItem>
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
          {selectedChart === "demographics" && renderDemographicsChart()}
          {selectedChart === "education" && renderEducationChart()}
          {selectedChart === "farming-experience" && renderFarmingExperienceChart()}
          {selectedChart === "land-ownership" && renderLandOwnershipChart()}
          {selectedChart === "varieties" && renderVarietiesChart()}
          {selectedChart === "regional" && renderRegionalChart()}
        </CardContent>
      </Card>

      {/* Statistics Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="hover:shadow-md transition-shadow duration-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Users className="w-4 h-4 text-green-600" />
              Total Petani
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
              <TrendingUp className="w-4 h-4 text-blue-600" />
              Rata-rata Umur
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{processedData.averageAge.toFixed(0)} tahun</div>
            <p className="text-xs text-muted-foreground">usia petani</p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow duration-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-amber-600" />
              Pengalaman Bertani
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-amber-600">{processedData.averageExperience.toFixed(0)} tahun</div>
            <p className="text-xs text-muted-foreground">rata-rata pengalaman</p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow duration-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <PieChartIcon className="w-4 h-4 text-purple-600" />
              Rata-rata Lahan
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600">{(processedData.averageLandSize / 1000).toFixed(1)}k m²</div>
            <p className="text-xs text-muted-foreground">luas lahan</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}