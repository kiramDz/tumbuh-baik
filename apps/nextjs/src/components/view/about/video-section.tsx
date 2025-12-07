"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import VideoPlayer from "@/components/ui/video-player";
import { Video } from "lucide-react";

export default function VideoSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Video className="h-5 w-5 text-purple-600" />
          Video Dokumentasi
        </CardTitle>
        <CardDescription>
          Dokumentasi visual tentang kearifan lokal dan praktik pertanian
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-8">
        {/* Video 1 */}
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold mb-1">Sistem Subak Bali</h3>
            <p className="text-sm text-muted-foreground">
              Dokumentasi tentang sistem irigasi tradisional Bali yang telah berusia ribuan tahun
            </p>
          </div>
          <VideoPlayer src="https://videos.pexels.com/video-files/3044127/3044127-uhd_2560_1440_25fps.mp4" />
        </div>

        <Separator />

        {/* Video 2 */}
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold mb-1">Pertanian Padi Organik</h3>
            <p className="text-sm text-muted-foreground">
              Proses penanaman padi organik dengan metode ramah lingkungan
            </p>
          </div>
          <VideoPlayer src="https://videos.pexels.com/video-files/30333849/13003128_2560_1440_25fps.mp4" />
        </div>

        <Separator />

        {/* Video 3 */}
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold mb-1">Tumpang Sari Modern</h3>
            <p className="text-sm text-muted-foreground">
              Implementasi sistem tumpang sari dengan pendekatan teknologi modern
            </p>
          </div>
          <VideoPlayer src="https://videos.pexels.com/video-files/7989489/7989489-uhd_2560_1440_25fps.mp4" />
        </div>
      </CardContent>
    </Card>
  );
}