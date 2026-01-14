"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export default function OverviewSection() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Tentang Tumbuh Baik</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground">
            Tumbuh Baik adalah platform digitalisasi pertanian yang menghubungkan petani dengan teknologi modern
            sambil tetap menghormati dan melestarikan kearifan lokal yang telah turun-temurun.
          </p>

          <p className="text-muted-foreground">
            Kami percaya bahwa teknologi dan tradisi dapat berjalan beriringan untuk menciptakan pertanian
            yang lebih produktif dan berkelanjutan.
          </p>
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Visi</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Menjadi platform pertanian digital terdepan yang mengintegrasikan teknologi dengan kearifan lokal
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Misi</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Memberdayakan petani melalui digitalisasi data dan pelestarian pengetahuan tradisional
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Nilai</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Kolaborasi, inovasi, dan keberlanjutan untuk masa depan pertanian Indonesia
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Apa yang Kami Lakukan</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-3 text-sm text-muted-foreground">
            <li className="flex gap-2">
              <span className="text-green-600">•</span>
              <span>Digitalisasi data pertanian untuk monitoring dan analisis yang lebih baik</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-600">•</span>
              <span>Dokumentasi kearifan lokal dalam bentuk digital yang mudah diakses</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-600">•</span>
              <span>Penyediaan informasi cuaca dan kalender tanam yang akurat</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-600">•</span>
              <span>Menghubungkan petani dengan pasar dan teknologi pertanian modern</span>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}