"use client";

import React from "react";
import { MapPin, Calendar, Globe, Flag } from "lucide-react";
import { useState, useEffect } from "react";
import Image from "next/image";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

export const Banner = React.memo(() => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const stats = [
    {
      icon: <Calendar className="w-4 h-4" />,
      value: currentTime.getFullYear().toString(),
      label: "Tahun",
      color: "text-gray-300",
      bg: "bg-gray-500/20",
      border: "border-gray-400/30"
    },
    {
      icon: <Flag className="w-4 h-4" />,
      value: "Indonesia",
      label: "Negara",
      color: "text-gray-300",
      bg: "bg-gray-500/20",
      border: "border-gray-400/30"
    },
    {
      icon: <Globe className="w-4 h-4" />,
      value: "Asia",
      label: "Benua",
      color: "text-gray-300",
      bg: "bg-gray-500/20",
      border: "border-gray-400/30"
    },
  ];

  return (
    <div className="relative mb-6 overflow-hidden">
      {/* Main Banner Container */}
      <div className="relative min-h-[350px] lg:min-h-[400px] rounded-xl overflow-hidden shadow-lg">
        
        {/* Background Image with Overlays */}
        <div className="absolute inset-0 h-full w-full">
          {/* Background Image - Optimized with Next.js Image */}
          <Image
            src="/image/aceh-besar-banner.jpg"
            alt="Aceh Besar Rice Field"
            fill
            priority
            quality={85}
            sizes="100vw"
            className="object-cover"
          />
          
          {/* Multi-layer Overlays for better readability */}
          <div className="absolute inset-0 bg-gradient-to-r from-black/50 via-black/30 to-transparent" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
        </div>

        {/* Main Content Layout */}
        <div className="relative h-full">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 lg:py-8">
            <div className="grid lg:grid-cols-12 gap-4 lg:gap-6 items-start min-h-[300px]">
              
              {/* Left Content - Main Information */}
              <div className="lg:col-span-8 space-y-4 min-h-[250px]">
                
                {/* Header Section */}
                <div>
                  <div className="flex items-start gap-3 mb-4">
                    <div className="p-2.5 bg-gray-500/20 rounded-lg backdrop-blur-sm border border-gray-400/30 shadow-lg">
                      <MapPin className="w-6 h-6 text-gray-300" />
                    </div>
                    <div className="flex-1">
                      <h1 className="text-3xl lg:text-4xl xl:text-5xl font-bold text-white mb-1 tracking-tight leading-tight">
                        Aceh Besar
                      </h1>
                      <p className="text-gray-200 text-base lg:text-lg font-medium">
                        {mounted ? (
                          currentTime.toLocaleDateString("id-ID", {
                            weekday: "long",
                            year: "numeric",
                            month: "long",
                            day: "numeric",
                          })
                        ) : (
                          <span className="opacity-0">Loading...</span>
                        )}
                      </p>
                      {/* Current Time Display */}
                      {mounted && (
                        <Badge variant="secondary" className="mt-2 text-sm font-mono bg-black/30 text-white border-white/20 backdrop-blur-sm hover:bg-black/40">
                          {currentTime.toLocaleTimeString("id-ID")}
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>

                {/* System Description Card */}
                <Card className="bg-white/10 backdrop-blur-xl border-white/20 hover:bg-white/15 transition-colors">
                  <CardHeader className="p-3 lg:p-4 pb-2">
                    <CardTitle className="text-white/90 text-base lg:text-lg">
                      Sistem Prakiraan Cuaca Terpadu
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-3 lg:p-4 pt-0">
                    <CardDescription className="text-white/70 text-sm lg:text-base">
                      Powered by BMKG and NASA Dataset
                    </CardDescription>
                  </CardContent>
                </Card>

              </div>

              {/* Right Sidebar - Stats - Fixed height to prevent CLS */}
              <div className="lg:col-span-4 min-h-[300px]">
                <Card className="bg-white/10 backdrop-blur-2xl border-white/20 transition-colors h-full">
                  <CardHeader className="p-4 pb-3">
                    <CardTitle className="text-white text-lg text-center">
                      Informasi Geografis
                    </CardTitle>
                  </CardHeader>
                  <Separator className="bg-white/20" />
                  <CardContent className="p-4 pt-4">
                    <div className="space-y-3">
                      {stats.map((stat, index) => (
                        <Card
                          key={index}
                          className={`bg-white/5 border transition-colors ${stat.border}`}
                        >
                          <CardContent className="p-3">
                            <div className="flex items-center gap-2.5">
                              <div className={`p-2 rounded-md ${stat.bg} border ${stat.border}`}>
                                <div className={stat.color}>
                                  {stat.icon}
                                </div>
                              </div>
                              <div className="leading-tight space-y-4">
                                <p className="text-xl font-bold text-white leading-none mb-0.5">
                                  {stat.value}
                                </p>
                                <p className="text-xs text-gray-300 leading-none">
                                  {stat.label}
                                </p>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

Banner.displayName = 'Banner';