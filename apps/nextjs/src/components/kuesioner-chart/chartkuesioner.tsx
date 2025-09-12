"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Users, Building2 } from "lucide-react";
import ChartKuesionerPetani from "./chartkuesion";
import ChartManajemen from "./chartmanajemen";

type DataType = "petani" | "manajemen";

export default function ChartKuesioner() {
  const [selectedDataType, setSelectedDataType] = useState<DataType>("petani");

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h2 className="text-2xl font-bold text-gray-900">Dashboard Kuesioner</h2>
          <p className="text-sm text-gray-600">Analisis data kuesioner petani dan manajemen usaha</p>
        </div>
      </div>

      <Separator />

      {/* Data Type Selection */}
      <Card className="bg-gradient-to-r from-green-50 to-blue-50">
        <CardContent className="p-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-gray-700">
              {selectedDataType === "petani" ? (
                <Users className="w-5 h-5 text-green-600" />
              ) : (
                <Building2 className="w-5 h-5 text-blue-600" />
              )}
              <span className="text-sm font-medium">Jenis Data:</span>
            </div>
            <Select value={selectedDataType} onValueChange={(value: DataType) => setSelectedDataType(value)}>
              <SelectTrigger className="w-64 bg-white">
                <SelectValue placeholder="Pilih jenis data" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="petani">
                  <div className="flex items-center gap-2">
                    <Users className="w-4 h-4 text-green-600" />
                    👨‍🌾 Data Petani
                  </div>
                </SelectItem>
                <SelectItem value="manajemen">
                  <div className="flex items-center gap-2">
                    <Building2 className="w-4 h-4 text-blue-600" />
                    🏢 Data Manajemen Usaha
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
            <Badge variant={selectedDataType === "petani" ? "default" : "secondary"} className="px-3 py-1">
              {selectedDataType === "petani" ? "Data Petani" : "Data Manajemen"}
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Render Selected Chart */}
      {selectedDataType === "petani" ? (
        <ChartKuesionerPetani />
      ) : (
        <ChartManajemen />
      )}
    </div>
  );
}