"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Users, Building2, Calendar } from "lucide-react";
import ChartKuesionerPetani from "./chartkuesion";
import ChartManajemen from "./chartmanajemen";
import ChartPeriode from "./chartperiode";

type DataType = "petani" | "manajemen" | "periode";

export default function ChartKuesioner() {
  const [selectedDataType, setSelectedDataType] = useState<DataType>("petani");

  return (
    <div className="space-y-6">
      <Card className="border-none shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-xl">Data Kuesioner</CardTitle>
          <CardDescription>
            Analisis hasil kuesioner dari berbagai aspek
          </CardDescription>
        </CardHeader>
      </Card>

      <Tabs value={selectedDataType} onValueChange={(value) => setSelectedDataType(value as DataType)} className="space-y-4">
        <Card className="border-none shadow-sm">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <TabsList>
                <TabsTrigger value="petani">
                  <Users className="w-4 h-4 mr-2" />
                  Petani
                </TabsTrigger>
                {/* <TabsTrigger value="manajemen">
                  <Building2 className="w-4 h-4 mr-2" />
                  Manajemen
                </TabsTrigger> */}
                {/* <TabsTrigger value="periode">
                  <Calendar className="w-4 h-4 mr-2" />
                  Periode Tanam
                </TabsTrigger> */}
              </TabsList>
            </div>
          </CardContent>
        </Card>

        <TabsContent value="petani">
          <Card>
            <CardHeader>
              <CardTitle>Data Petani</CardTitle>
              <CardDescription>
                Informasi demografi dan karakteristik petani responden
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ChartKuesionerPetani />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="manajemen">
          <Card>
            <CardHeader>
              <CardTitle>Manajemen Usaha Tani</CardTitle>
              <CardDescription>
                Data pengelolaan dan praktik usaha tani
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ChartManajemen />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="periode">
          <Card>
            <CardHeader>
              <CardTitle>Periode Tanam</CardTitle>
              <CardDescription>
                Informasi waktu tanam dan pola musim
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ChartPeriode />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}