"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface FarmData {
  luasTanam: number;
  hasilPanen: number;
  biaya: number;
  keuntungan: number;
  [key: string]: any;
}

interface StatistikOverviewProps {
  farmData: FarmData;
}

export default function StatistikOverview({ farmData }: StatistikOverviewProps) {
  const pendapatanKotor = farmData.biaya + farmData.keuntungan;
  const marginKeuntungan = pendapatanKotor > 0 ? (farmData.keuntungan / pendapatanKotor) * 100 : 0;
  const biayaPerHa = farmData.luasTanam > 0 ? farmData.biaya / farmData.luasTanam : 0;
  const pendapatanPerHa = farmData.luasTanam > 0 ? pendapatanKotor / farmData.luasTanam : 0;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Financial Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Analisis Finansial</CardTitle>
          <CardDescription>Breakdown pendapatan dan biaya produksi</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Biaya Produksi</span>
              <span className="font-medium text-red-600">
                Rp {farmData.biaya.toLocaleString('id-ID')}
              </span>
            </div>
            <Progress 
              value={(farmData.biaya / pendapatanKotor) * 100} 
              className="h-2" 
            />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Keuntungan</span>
              <span className="font-medium text-green-600">
                Rp {farmData.keuntungan.toLocaleString('id-ID')}
              </span>
            </div>
            <Progress 
              value={marginKeuntungan} 
              className="h-2" 
            />
          </div>

          <div className="pt-2 border-t">
            <div className="flex justify-between">
              <span className="font-medium">Total Pendapatan</span>
              <span className="font-bold text-blue-600">
                Rp {pendapatanKotor.toLocaleString('id-ID')}
              </span>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Margin keuntungan: {marginKeuntungan.toFixed(1)}%
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Productivity Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Analisis Produktivitas</CardTitle>
          <CardDescription>Efisiensi penggunaan lahan per hektar</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-3 bg-blue-50 rounded-lg">
              <p className="text-2xl font-bold text-blue-600">
                {(farmData.hasilPanen / farmData.luasTanam).toFixed(2)}
              </p>
              <p className="text-xs text-gray-600">Ton/Ha</p>
              <p className="text-xs text-gray-500">Produktivitas</p>
            </div>
            
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <p className="text-2xl font-bold text-green-600">
                Rp {(pendapatanPerHa / 1000000).toFixed(1)}M
              </p>
              <p className="text-xs text-gray-600">per Ha</p>
              <p className="text-xs text-gray-500">Pendapatan</p>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Biaya per Ha:</span>
              <span className="text-sm font-medium">
                Rp {biayaPerHa.toLocaleString('id-ID')}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Keuntungan per Ha:</span>
              <span className="text-sm font-medium text-green-600">
                Rp {(farmData.keuntungan / farmData.luasTanam).toLocaleString('id-ID')}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">ROI per Ha:</span>
              <span className="text-sm font-medium">
                {((farmData.keuntungan / farmData.luasTanam) / biayaPerHa * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Planting Details */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Detail Penanaman</CardTitle>
          <CardDescription>Informasi lengkap tentang siklus tanam</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{farmData.luasTanam}</div>
              <div className="text-sm text-gray-600">Hektar</div>
              <div className="text-xs text-gray-500">Luas Lahan</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{farmData.hasilPanen}</div>
              <div className="text-sm text-gray-600">Ton</div>
              <div className="text-xs text-gray-500">Total Panen</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {(farmData.keuntungan / farmData.biaya * 100).toFixed(0)}%
              </div>
              <div className="text-sm text-gray-600">ROI</div>
              <div className="text-xs text-gray-500">Return on Investment</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {Math.ceil(farmData.luasTanam * 90)}
              </div>
              <div className="text-sm text-gray-600">Hari</div>
              <div className="text-xs text-gray-500">Siklus Tanam</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}