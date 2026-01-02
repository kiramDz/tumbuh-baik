"use client";

import React from "react";
import Image from "next/image";
import { Cloud } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "../ui/badge";
import { LineShadowText } from "../ui/line-shadow-text";
import { WeatherConclusionResult } from "@/lib/bmkg-utils";

interface WeatherConclusionProps {
  conclusion: WeatherConclusionResult & {
    seasonalStatus: string;
  };
  tcc: number;
}


const WeatherConclusion: React.FC<WeatherConclusionProps> = ({ conclusion, tcc }) => {
  const isCocok = conclusion.status === "cocok" && conclusion.seasonalStatus !== "tidak cocok tanam";
  const isWaspada = conclusion.status === "cocok" && conclusion.seasonalStatus === "tidak cocok tanam";

  const title = isCocok ? "Cocok" : isWaspada ? "Waspada" : "Tidak Dianjurkan";
  const subtitle = conclusion.reason;

  const getBackgroundImage = (cloud: number) => {
    if (cloud >= 70) return "/image/rainy.png";
    if (cloud >= 30) return "/image/cloudy.png";
    return "/image/sunny.png";
  };
  

  return (
    <Card className="relative overflow-hidden">
      {/* Optimized background image using Next.js Image */}
      <div className="absolute inset-0">
        <Image
          src={getBackgroundImage(tcc)}
          alt={`Weather background for ${title}`}
          fill
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          className="object-cover"
          priority={false}
          quality={75}
        />
      </div>
      <CardHeader className="pb-0 relative z-10">
        <CardTitle className="flex items-center gap-2">
          <Badge variant="secondary" className="bg-[#344a53] text-white dark:bg-blue-600 rounded-full">
            <Cloud className="h-4 w-4" />
            {title}
          </Badge>
        </CardTitle>
      </CardHeader>
      <div className="w-full z-10  h-full relative flex flex-col justify-center">
        <CardContent className=" flex flex-col items-center justify-between">
          <h1 className="text-balance text-5xl font-semibold leading-none tracking-tighter sm:text-6xl md:text-7xl ">
            {title}
            <LineShadowText className="italic" shadowColor="white">
              Tanam
            </LineShadowText>
          </h1>
          {/* mempertimbangakn data, buat kesimpulan */}
          <p>{subtitle}</p>
          <div className="flex flex-wrap gap-2 mt-2">
            {conclusion.badge.map((b, i) => (
              <Badge key={i} variant="outline">
                {b}
              </Badge>
            ))}
          </div>
        </CardContent>
      </div>
    </Card>
  );
};

export default WeatherConclusion;
