"use client";

import React, { useState, useMemo } from "react";
import { MapPin, Clock, Search, Calendar } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { 
  Select, 
  SelectContent, 
  SelectGroup,
  SelectItem, 
  SelectLabel,
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import type { BMKGApiData } from "@/types/table-schema";
import { getUniqueGampongData } from "@/lib/bmkg-utils";

interface WeatherHeaderProps {
  bmkgData: BMKGApiData[];
  selectedCode: string;
  onGampongChange: (code: string) => void;
}

export const WeatherHeader = React.memo<WeatherHeaderProps>(({ 
  bmkgData, 
  selectedCode, 
  onGampongChange 
}) => {
  const [searchQuery, setSearchQuery] = useState("");

  const uniqueGampongs = getUniqueGampongData(bmkgData);
  const selected = uniqueGampongs.find((item) => item.kode_gampong === selectedCode);
  const tanggal = selected?.tanggal_data ? new Date(selected.tanggal_data) : new Date();

  // Date formatting
  const day = tanggal.toLocaleDateString("id-ID", { weekday: "long" });
  const date = tanggal.getDate();
  const month = tanggal.toLocaleDateString("id-ID", { month: "long" });
  const year = tanggal.getFullYear();
  const time = tanggal.toLocaleTimeString("id-ID", { hour: '2-digit', minute: '2-digit' });

  // Filter locations based on search
  const filteredLocations = useMemo(() => {
    if (!searchQuery) return uniqueGampongs;
    return uniqueGampongs.filter((item) => 
      item.nama_gampong.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [uniqueGampongs, searchQuery]);



  return (
    <Card className="bg-white dark:bg-gray-950 border shadow-sm overflow-hidden">

      <CardHeader className="relative pb-4">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
          
          {/* Left Section - Date & Location Info */}
          <div className="flex items-center gap-5">
            
            {/* Date Card */}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-br from-gray-700 to-gray-600 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity" />
              <div className="relative bg-gradient-to-br from-gray-800 to-gray-700 p-4 rounded-2xl shadow-lg border border-white/20">
                <div className="flex flex-col items-center">
                  <Calendar className="w-5 h-5 text-white/80 mb-1" />
                  <span className="text-4xl font-bold text-white leading-none">{date}</span>
                  <span className="text-xs font-medium text-white/90 mt-1 uppercase tracking-wider">{month.slice(0, 3)}</span>
                </div>
              </div>
            </div>

            {/* Date & Location Info */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium text-muted-foreground">
                  {day}, {date} {month} {year}
                </span>
              </div>
              
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4 text-foreground" />
                <span className="text-lg font-semibold text-foreground">
                  {selected?.nama_gampong || "Pilih Lokasi"}
                </span>
              </div>
            </div>
          </div>

          {/* Right Section - Location Selector */}
          <div className="hidden flex items-center gap-3 w-full lg:w-auto">
            
            {/* Location Selector */}
            <Select value={selectedCode} onValueChange={onGampongChange}>
              <SelectTrigger className="w-full lg:w-[320px] h-11">
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4" />
                  <SelectValue placeholder="Pilih Lokasi" />
                </div>
              </SelectTrigger>
              
              <SelectContent className="w-[320px]">
                {/* Search Box */}
                <div className="p-3 border-b">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      placeholder="Cari lokasi..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-9 h-9"
                    />
                  </div>
                </div>

                {/* All Locations Section */}
                <SelectGroup>
                  <div className="flex items-center gap-2 px-3 py-2 bg-muted/50">
                    <MapPin className="w-3.5 h-3.5 text-muted-foreground" />
                    <SelectLabel className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                      Semua Lokasi ({filteredLocations.length})
                    </SelectLabel>
                  </div>
                  {filteredLocations.length > 0 ? (
                    filteredLocations.map((item, index) => (
                      <SelectItem 
                        key={`${item.kode_gampong}-${index}`} 
                        value={item.kode_gampong}
                        className="cursor-pointer"
                      >
                        <span>{item.nama_gampong}</span>
                      </SelectItem>
                    ))
                  ) : (
                    <div className="px-4 py-8 text-center">
                      <MapPin className="w-10 h-10 text-muted-foreground mx-auto mb-2" />
                      <p className="text-sm text-muted-foreground">
                        Lokasi tidak ditemukan
                      </p>
                    </div>
                  )}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>

      {/* Bottom Accent Line */}
      <div className="h-1 bg-gradient-to-r from-gray-700 via-gray-600 to-gray-500" />
    </Card>
  );
});

WeatherHeader.displayName = 'WeatherHeader';
