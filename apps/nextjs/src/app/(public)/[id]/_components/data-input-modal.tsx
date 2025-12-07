"use client";

import { useState } from "react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Sprout, MapPin, Calendar, TrendingUp, DollarSign, Package } from "lucide-react";

interface FarmData {
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
}

interface DataInputModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: Partial<FarmData>) => void;
  initialData?: FarmData | null;
  farmId: string;
}

export default function DataInputModal({ 
  isOpen, 
  onClose, 
  onSave, 
  initialData, 
  farmId 
}: DataInputModalProps) {
  const [formData, setFormData] = useState({
    name: initialData?.name || `Farm ${farmId}`,
    location: initialData?.location || "",
    luasTanam: initialData?.luasTanam || 0,
    hasilPanen: initialData?.hasilPanen || 0,
    biaya: initialData?.biaya || 0,
    keuntungan: initialData?.keuntungan || 0,
    season: initialData?.season || "Musim Tanam 2024",
    plantingDate: initialData?.plantingDate || new Date().toISOString().split('T')[0],
    harvestDate: initialData?.harvestDate || new Date().toISOString().split('T')[0],
  });

  const handleInputChange = (field: string, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      toast.error("Nama farm harus diisi");
      return;
    }
    
    if (!formData.location.trim()) {
      toast.error("Lokasi farm harus diisi");
      return;
    }
    
    if (formData.luasTanam <= 0) {
      toast.error("Luas tanam harus lebih dari 0");
      return;
    }

    if (formData.hasilPanen < 0) {
      toast.error("Hasil panen tidak boleh negatif");
      return;
    }

    if (formData.biaya < 0) {
      toast.error("Biaya tidak boleh negatif");
      return;
    }

    const processedData = {
      ...formData,
      luasTanam: Number(formData.luasTanam),
      hasilPanen: Number(formData.hasilPanen),
      biaya: Number(formData.biaya),
      keuntungan: Number(formData.keuntungan),
    };

    onSave(processedData);
    toast.success(initialData ? "Data berhasil diupdate!" : "Data berhasil disimpan!");
  };

  const produktivitas = formData.luasTanam > 0 ? (formData.hasilPanen / formData.luasTanam).toFixed(2) : '0';
  const roi = formData.biaya > 0 ? ((formData.keuntungan / formData.biaya) * 100).toFixed(1) : '0';
  const marginKeuntungan = (formData.biaya + formData.keuntungan) > 0 
    ? ((formData.keuntungan / (formData.biaya + formData.keuntungan)) * 100).toFixed(1) 
    : '0';

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto rounded-2xl">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-xl">
              <Sprout className="h-5 w-5 text-primary" />
            </div>
            <div className="space-y-2">
              <DialogTitle className="text-xl font-semibold leading-tight">
                {initialData ? `Edit Data Farm` : `Setup Farm Baru`}
              </DialogTitle>
              <DialogDescription className="leading-tight">
                {initialData 
                  ? "Update informasi dan data pertanian" 
                  : "Isi data pertanian untuk analisis dashboard"}
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-6 mt-4">
          
          {/* Informasi Farm */}
          <Card className="rounded-xl border-2">
            <CardContent className="pt-6 space-y-4">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-1.5 bg-muted rounded-lg">
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                </div>
                <h3 className="font-semibold">Informasi Farm</h3>
              </div>
              
              <div className="grid gap-4">
                <div className="grid sm:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="name" className="text-sm font-medium">Nama Farm</Label>
                    <Input
                      id="name"
                      placeholder="Green Valley Farm"
                      value={formData.name}
                      onChange={(e) => handleInputChange('name', e.target.value)}
                      className="rounded-lg"
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="location" className="text-sm font-medium">Lokasi</Label>
                    <Input
                      id="location"
                      placeholder="Malang, Jawa Timur"
                      value={formData.location}
                      onChange={(e) => handleInputChange('location', e.target.value)}
                      className="rounded-lg"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="season" className="text-sm font-medium">Musim Tanam</Label>
                  <Input
                    id="season"
                    placeholder="Musim Kemarau 2024"
                    value={formData.season}
                    onChange={(e) => handleInputChange('season', e.target.value)}
                    className="rounded-lg"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Data Pertanian */}
          <Card className="rounded-xl border-2">
            <CardContent className="pt-6 space-y-4">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-1.5 bg-muted rounded-lg">
                  <Package className="h-4 w-4 text-muted-foreground" />
                </div>
                <h3 className="font-semibold">Data Pertanian</h3>
              </div>
              
              <div className="grid gap-4">
                <div className="grid sm:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="luasTanam" className="text-sm font-medium">Luas Tanam (Ha)</Label>
                    <Input
                      id="luasTanam"
                      type="number"
                      step="0.01"
                      min="0"
                      placeholder="2.5"
                      value={formData.luasTanam || ''}
                      onChange={(e) => handleInputChange('luasTanam', parseFloat(e.target.value) || 0)}
                      className="rounded-lg"
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="hasilPanen" className="text-sm font-medium">Hasil Panen (Ton)</Label>
                    <Input
                      id="hasilPanen"
                      type="number"
                      step="0.01"
                      min="0"
                      placeholder="15.2"
                      value={formData.hasilPanen || ''}
                      onChange={(e) => handleInputChange('hasilPanen', parseFloat(e.target.value) || 0)}
                      className="rounded-lg"
                      required
                    />
                  </div>
                </div>

                <div className="grid sm:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="biaya" className="text-sm font-medium">Biaya (Rp)</Label>
                    <Input
                      id="biaya"
                      type="number"
                      min="0"
                      placeholder="25000000"
                      value={formData.biaya || ''}
                      onChange={(e) => handleInputChange('biaya', parseInt(e.target.value) || 0)}
                      className="rounded-lg"
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="keuntungan" className="text-sm font-medium">Keuntungan (Rp)</Label>
                    <Input
                      id="keuntungan"
                      type="number"
                      min="0"
                      placeholder="40000000"
                      value={formData.keuntungan || ''}
                      onChange={(e) => handleInputChange('keuntungan', parseInt(e.target.value) || 0)}
                      className="rounded-lg"
                      required
                    />
                  </div>
                </div>

                <div className="grid sm:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="plantingDate" className="text-sm font-medium">Tanggal Tanam</Label>
                    <Input
                      id="plantingDate"
                      type="date"
                      value={formData.plantingDate}
                      onChange={(e) => handleInputChange('plantingDate', e.target.value)}
                      className="rounded-lg"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="harvestDate" className="text-sm font-medium">Tanggal Panen</Label>
                    <Input
                      id="harvestDate"
                      type="date"
                      value={formData.harvestDate}
                      onChange={(e) => handleInputChange('harvestDate', e.target.value)}
                      className="rounded-lg"
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Preview */}
          {(formData.luasTanam > 0 || formData.hasilPanen > 0) && (
            <Card className="rounded-xl border-0 bg-muted/30">
              <CardContent className="pt-6">
                <div className="flex items-center gap-2 mb-4">
                  <div className="p-1.5 bg-background rounded-lg">
                    <TrendingUp className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <h3 className="font-semibold">Preview Perhitungan</h3>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <Card className="rounded-xl border-0 shadow-sm">
                    <CardContent className="pt-6 pb-4 text-center">
                      <div className="text-2xl font-bold mb-1">{produktivitas}</div>
                      <Badge variant="secondary" className="rounded-full text-xs">
                        Ton/Ha
                      </Badge>
                    </CardContent>
                  </Card>
                  
                  <Card className="rounded-xl border-0 shadow-sm">
                    <CardContent className="pt-6 pb-4 text-center">
                      <div className="text-2xl font-bold mb-1">{roi}%</div>
                      <Badge variant="secondary" className="rounded-full text-xs">
                        ROI
                      </Badge>
                    </CardContent>
                  </Card>
                  
                  <Card className="rounded-xl border-0 shadow-sm">
                    <CardContent className="pt-6 pb-4 text-center">
                      <div className="text-2xl font-bold mb-1">{marginKeuntungan}%</div>
                      <Badge variant="secondary" className="rounded-full text-xs">
                        Margin
                      </Badge>
                    </CardContent>
                  </Card>
                </div>
              </CardContent>
            </Card>
          )}

          <div className="flex justify-end gap-3 pt-4">
            <Button type="button" variant="outline" onClick={onClose} className="rounded-lg">
              Batal
            </Button>
            <Button type="submit" className="rounded-lg">
              {initialData ? "Update Data" : "Simpan Data"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}