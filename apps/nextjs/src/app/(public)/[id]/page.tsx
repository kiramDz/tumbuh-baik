"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { useSession } from "@/lib/better-auth/auth-client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { 
  BarChart3, 
  TrendingUp, 
  DollarSign, 
  Sprout, 
  Edit, 
  Calendar,
  Users,
  MapPin
} from "lucide-react";
import StatistikOverview from "./_components/statistik-overview";
import ProductivityAnalysis from "./_components/productivity_analysis";
import SeasonalComparison from "./_components/seasonal-comparison";
import DataInputModal from "./_components/data-input-modal";

// Import API functions
import { 
  getFarmById, 
  createFarm, 
  updateFarm,
  getFarmsByUserId
} from "@/lib/fetch/files.fetch";

const DashboardCardSkeleton = () => (
  <Card>
    <CardHeader className="pb-3">
      <Skeleton className="h-4 w-24" />
    </CardHeader>
    <CardContent>
      <Skeleton className="h-7 w-16 mb-2" />
      <Skeleton className="h-3 w-20" />
    </CardContent>
  </Card>
);

interface FarmData {
  _id?: string; // Tambahkan _id dari MongoDB
  id: string;
  name: string;
  location: string;
  owner: string;
  luasTanam: number;
  hasilPanen: number;
  biaya: number;
  keuntungan: number;
  season: string;
  plantingDate: string;
  harvestDate: string;
  lastUpdated: string;
  userId?: string;
}

export default function FarmDashboardPage() {
  const params = useParams();
  const farmId = params.id as string;
  const { data: session } = useSession();
  
  const [farmData, setFarmData] = useState<FarmData | null>(null);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isFirstTime, setIsFirstTime] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (session?.user) {
      loadFarmData();
    }
  }, [farmId, session?.user]);

  const loadFarmData = async () => {
    if (!session?.user?.id) {
      setLoading(false);
      return;
    }

    try {
      // Gunakan getFarmsByUserId untuk ambil farm milik user yang login
      const farms = await getFarmsByUserId(session.user.id);
      
      if (farms && farms.length > 0) {
        // Ambil farm pertama user (atau bisa filter berdasarkan farmId)
        const data = farms[0]; // atau farms.find(f => f.customId === farmId)
        
        // Format data dari MongoDB
        const formattedData: FarmData = {
          _id: data._id,
          id: data._id, // Gunakan _id sebagai id
          name: data.name,
          location: data.location,
          owner: session.user.name || "Owner",
          luasTanam: data.luasTanam,
          hasilPanen: data.hasilPanen,
          biaya: data.biaya,
          keuntungan: data.keuntungan,
          season: data.season,
          plantingDate: data.plantingDate.split('T')[0],
          harvestDate: data.harvestDate.split('T')[0],
          lastUpdated: data.updatedAt,
          userId: data.userId
        };
        
        setFarmData(formattedData);
        setIsFirstTime(false);
      } else {
        // Tidak ada farm untuk user ini
        setIsFirstTime(true);
        setIsEditModalOpen(true);
      }
    } catch (error: any) {
      console.error("Error loading farm data:", error);
      
      // Jika error atau data tidak ada
      setIsFirstTime(true);
      setIsEditModalOpen(true);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveData = async (data: Partial<FarmData>) => {
    if (!session?.user?.id) {
      toast.error("User tidak terautentikasi");
      return;
    }

    try {
      setLoading(true);

      const farmPayload = {
        name: data.name || `Farm ${farmId}`,
        location: data.location || "",
        luasTanam: data.luasTanam || 0,
        hasilPanen: data.hasilPanen || 0,
        biaya: data.biaya || 0,
        keuntungan: data.keuntungan || 0,
        season: data.season || "Musim Tanam 2024",
        plantingDate: data.plantingDate || new Date().toISOString().split('T')[0],
        harvestDate: data.harvestDate || new Date().toISOString().split('T')[0],
        userId: session.user.id
      };

      let savedData;

      if (farmData?._id) {
        // Update existing farm
        savedData = await updateFarm(farmData._id, farmPayload);
        toast.success("Data farm berhasil diupdate!");
      } else {
        // Create new farm
        savedData = await createFarm(farmPayload);
        toast.success("Data farm berhasil disimpan!");
      }

      // Format data untuk state
      const formattedData: FarmData = {
        _id: savedData._id,
        id: farmId,
        name: savedData.name,
        location: savedData.location,
        owner: session.user.name || "Owner",
        luasTanam: savedData.luasTanam,
        hasilPanen: savedData.hasilPanen,
        biaya: savedData.biaya,
        keuntungan: savedData.keuntungan,
        season: savedData.season,
        plantingDate: savedData.plantingDate.split('T')[0],
        harvestDate: savedData.harvestDate.split('T')[0],
        lastUpdated: savedData.updatedAt,
        userId: savedData.userId
      };

      setFarmData(formattedData);
      setIsEditModalOpen(false);
      setIsFirstTime(false);
      
    } catch (error) {
      console.error("Error saving farm data:", error);
      toast.error("Gagal menyimpan data farm");
    } finally {
      setLoading(false);
    }
  };

  // Show loading while waiting for session
  if (!session) {
    return (
      <main className="p-6 max-w-7xl mx-auto">
        <div className="space-y-6">
          <Skeleton className="h-8 w-48" />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <DashboardCardSkeleton key={i} />
            ))}
          </div>
        </div>
      </main>
    );
  }

  if (loading) {
    return (
      <main className="p-6 max-w-7xl mx-auto">
        <div className="space-y-6">
          <Skeleton className="h-8 w-48" />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <DashboardCardSkeleton key={i} />
            ))}
          </div>
        </div>
      </main>
    );
  }

  if (isFirstTime && !farmData) {
    return (
      <>
        <main className="flex items-center justify-center min-h-screen p-6">
          <Card className="max-w-md w-full">
            <CardHeader>
              <CardTitle>Setup Farm Dashboard</CardTitle>
              <CardDescription>
                Belum ada data untuk farm ini. Silakan isi data terlebih dahulu.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={() => setIsEditModalOpen(true)} 
                className="w-full"
                size="lg"
              >
                <Sprout className="h-4 w-4 mr-2" />
                Mulai Input Data
              </Button>
            </CardContent>
          </Card>
        </main>
        <DataInputModal
          isOpen={isEditModalOpen}
          onClose={() => setIsEditModalOpen(false)}
          onSave={handleSaveData}
          initialData={farmData}
          farmId={farmId}
        />
      </>
    );
  }

  if (!farmData) return null;

  const produktivitas = farmData.luasTanam > 0 ? farmData.hasilPanen / farmData.luasTanam : 0;
  const roi = farmData.biaya > 0 ? (farmData.keuntungan / farmData.biaya) * 100 : 0;

  return (
    <>
      <main className="p-6 max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div className="space-y-1">
            <h1 className="text-3xl font-bold tracking-tight">{farmData.name}</h1>
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-1">
                <MapPin className="h-4 w-4" />
                {farmData.location}
              </div>
              <div className="flex items-center gap-1">
                <Users className="h-4 w-4" />
                {farmData.owner}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="text-sm">
              <Calendar className="h-3 w-3 mr-1" />
              {farmData.season}
            </Badge>
            <Button onClick={() => setIsEditModalOpen(true)} size="sm">
              <Edit className="h-4 w-4 mr-2" />
              Edit Data
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium">Area</CardTitle>
                <Sprout className="h-4 w-4 text-muted-foreground" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{farmData.luasTanam} Ha</div>
              <p className="text-xs text-muted-foreground mt-1">Total planted area</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium">Productivity</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{produktivitas.toFixed(2)} T/Ha</div>
              <p className="text-xs text-muted-foreground mt-1">Total: {farmData.hasilPanen} Ton</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium">ROI</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{roi.toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground mt-1">Return on investment</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium">Profit</CardTitle>
                <DollarSign className="h-4 w-4 text-muted-foreground" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                Rp {(farmData.keuntungan / 1000000).toFixed(1)}M
              </div>
              <p className="text-xs text-muted-foreground mt-1">Net profit</p>
            </CardContent>
          </Card>
        </div>

        {/* Tabs */}
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="productivity">Productivity</TabsTrigger>
            <TabsTrigger value="seasonal">Seasonal</TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <StatistikOverview farmData={farmData} />
          </TabsContent>

          <TabsContent value="productivity">
            <ProductivityAnalysis farmData={farmData} />
          </TabsContent>

          <TabsContent value="seasonal">
            <SeasonalComparison farmData={farmData} />
          </TabsContent>
        </Tabs>

        <DataInputModal
          isOpen={isEditModalOpen}
          onClose={() => setIsEditModalOpen(false)}
          onSave={handleSaveData}
          initialData={farmData}
          farmId={farmId}
        />
      </main>
    </>
  );
}