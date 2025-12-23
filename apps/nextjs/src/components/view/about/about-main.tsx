"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BookOpen, Heart, Video, Users } from "lucide-react";
import OverviewSection from "@/components/view/about/overview-section";
import LocalWisdomSection from "@/components/view/about/local-wisdom-section";
import VideoSection from "@/components/view/about/video-section";
import TeamSection from "@/components/view/about/team-section";

function AboutContent() {
  return (
    <div className="container mx-auto py-8 px-4 max-w-7xl">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">Tentang Tumbuh Baik</h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Platform digitalisasi pertanian yang menggabungkan teknologi modern dengan kearifan lokal
        </p>
      </div>

      <Tabs defaultValue="overview" className="space-y-8">
        <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-grid">
          <TabsTrigger value="overview" className="gap-2">
            <BookOpen className="h-4 w-4" />
            <span className="hidden sm:inline">Gambaran</span>
          </TabsTrigger>
          <TabsTrigger value="local-wisdom" className="gap-2">
            <Heart className="h-4 w-4" />
            <span className="hidden sm:inline">Kearifan Lokal</span>
          </TabsTrigger>
          <TabsTrigger value="video" className="gap-2">
            <Video className="h-4 w-4" />
            <span className="hidden sm:inline">Video</span>
          </TabsTrigger>
          <TabsTrigger value="team" className="gap-2">
            <Users className="h-4 w-4" />
            <span className="hidden sm:inline">Tim</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <OverviewSection />
        </TabsContent>

        <TabsContent value="local-wisdom">
          <LocalWisdomSection />
        </TabsContent>

        <TabsContent value="video">
          <VideoSection />
        </TabsContent>

        <TabsContent value="team">
          <TeamSection />
        </TabsContent>
      </Tabs>
    </div>
  );
}

export { AboutContent };