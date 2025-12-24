"use client";

import React, { useState, useMemo } from "react";
import { MapPin, Clock, Search, Star, Calendar } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { BMKGApiData } from "@/types/table-schema";
import { getUniqueGampongData } from "@/lib/bmkg-utils";

interface WeatherHeaderProps {
  bmkgData: BMKGApiData[];
  selectedCode: string;
  onGampongChange: (code: string) => void;
}

export const WeatherHeader: React.FC<WeatherHeaderProps> = ({ bmkgData, selectedCode, onGampongChange }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [favorites, setFavorites] = useState<string[]>([]);

  const uniqueGampongs = getUniqueGampongData(bmkgData);
  const selected = uniqueGampongs.find((item) => item.kode_gampong === selectedCode);
  const tanggal = selected?.tanggal_data ? new Date(selected.tanggal_data) : new Date();

  // Date formatting
  const day = tanggal.toLocaleDateString("id-ID", { weekday: "long" });
  const date = tanggal.getDate();
  const month = tanggal.toLocaleDateString("id-ID", { month: "long" });
  const year = tanggal.getFullYear();
  const time = tanggal.toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit" });

  // Filter locations based on search
  const filteredLocations = useMemo(() => {
    if (!searchQuery) return uniqueGampongs;
    return uniqueGampongs.filter((item) => item.nama_gampong.toLowerCase().includes(searchQuery.toLowerCase()));
  }, [uniqueGampongs, searchQuery]);

  // Separate favorites and non-favorites
  const { favoriteLocations, nonFavoriteLocations } = useMemo(() => {
    const favs = filteredLocations.filter((item) => favorites.includes(item.kode_gampong));
    const nonFavs = filteredLocations.filter((item) => !favorites.includes(item.kode_gampong));
    return { favoriteLocations: favs, nonFavoriteLocations: nonFavs };
  }, [filteredLocations, favorites]);

  // Toggle favorite
  const toggleFavorite = (code: string) => {
    setFavorites((prev) => (prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code]));
  };

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
                <span className="text-lg font-semibold text-foreground">Aceh Besar</span>

                {/* Favorite Button */}
              </div>
            </div>
          </div>

          {/* Right Section - Location Selector */}
          <div className="flex items-center gap-3 w-full lg:w-auto">
            {/* Favorites Badge (if any) */}
            {favorites.length > 0 && (
              <Badge variant="outline" className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950 text-yellow-700 dark:text-yellow-400 hidden md:flex items-center gap-1.5">
                <Star className="w-3 h-3 fill-yellow-400" />
                {favorites.length} Favorit
              </Badge>
            )}

            {/* Location Selector */}
            {/* <Select value={selectedCode} onValueChange={onGampongChange}>
              <SelectTrigger className="w-full lg:w-[320px] h-11">
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4" />
                  <SelectValue placeholder="Pilih Lokasi" />
                </div>
              </SelectTrigger>

              <SelectContent className="w-[320px]">
                <div className="p-3 border-b">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input placeholder="Cari lokasi..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="pl-9 h-9" />
                  </div>
                </div>

                {favoriteLocations.length > 0 && (
                  <SelectGroup>
                    <div className="flex items-center gap-2 px-3 py-2 bg-yellow-50 dark:bg-yellow-950">
                      <Star className="w-3.5 h-3.5 fill-yellow-400 text-yellow-400" />
                      <SelectLabel className="text-xs font-semibold text-yellow-700 dark:text-yellow-400 uppercase tracking-wide">Lokasi Favorit</SelectLabel>
                    </div>
                    {favoriteLocations.map((item) => (
                      <SelectItem key={`fav-${item.kode_gampong}`} value={item.kode_gampong} className="pl-10 hover:bg-yellow-50 dark:hover:bg-yellow-950 cursor-pointer">
                        <div className="flex items-center gap-2">
                          <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                          <span className="font-medium">{item.nama_gampong}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectGroup>
                )}

                <SelectGroup>
                  <div className="flex items-center gap-2 px-3 py-2 bg-muted/50">
                    <MapPin className="w-3.5 h-3.5 text-muted-foreground" />
                    <SelectLabel className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                      {favoriteLocations.length > 0 ? "Lokasi Lainnya" : "Semua Lokasi"} ({nonFavoriteLocations.length})
                    </SelectLabel>
                  </div>
                  {nonFavoriteLocations.length > 0 ? (
                    nonFavoriteLocations.map((item, index) => (
                      <SelectItem key={`${item.kode_gampong}-${index}`} value={item.kode_gampong} className="cursor-pointer">
                        <div className="flex items-center justify-between w-full group">
                          <span>{item.nama_gampong}</span>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleFavorite(item.kode_gampong);
                            }}
                          >
                            <Star className="w-3 h-3 text-muted-foreground hover:text-yellow-400" />
                          </Button>
                        </div>
                      </SelectItem>
                    ))
                  ) : (
                    <div className="px-4 py-8 text-center">
                      <MapPin className="w-10 h-10 text-muted-foreground mx-auto mb-2" />
                      <p className="text-sm text-muted-foreground">{searchQuery ? "Lokasi tidak ditemukan" : "Semua lokasi sudah difavoritkan"}</p>
                    </div>
                  )}
                </SelectGroup>
              </SelectContent>
            </Select> */}
          </div>
        </div>
      </CardHeader>

      {/* Bottom Accent Line */}
      <div className="h-1 bg-gradient-to-r from-gray-700 via-gray-600 to-gray-500" />
    </Card>
  );
};
