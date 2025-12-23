"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Heart } from "lucide-react";

export default function LocalWisdomSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Heart className="h-5 w-5 text-red-600" />
          Kearifan Lokal Pertanian
        </CardTitle>
        <CardDescription>
          Pengetahuan tradisional yang diwariskan turun-temurun
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Pranata Mangsa */}
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <Badge variant="secondary" className="mt-1">Jawa</Badge>
            <div className="flex-1">
              <h3 className="font-semibold mb-2">Pranata Mangsa</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Sistem penanggalan tradisional Jawa yang mengatur waktu tanam berdasarkan perubahan musim. 
                Terdiri dari 12 mangsa (periode) yang masing-masing memiliki karakteristik cuaca dan 
                rekomendasi aktivitas pertanian tertentu.
              </p>
              <div className="mt-3 grid sm:grid-cols-2 gap-2">
                <div className="text-xs p-2 bg-green-50 rounded">
                  <span className="font-medium">Kasa:</span> Musim kemarau awal
                </div>
                <div className="text-xs p-2 bg-blue-50 rounded">
                  <span className="font-medium">Katiga:</span> Musim hujan mulai
                </div>
              </div>
            </div>
          </div>
        </div>

        <Separator />

        {/* Subak */}
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <Badge variant="secondary" className="mt-1">Bali</Badge>
            <div className="flex-1">
              <h3 className="font-semibold mb-2">Sistem Subak</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Organisasi irigasi tradisional Bali yang mengatur pembagian air secara adil berdasarkan 
                filosofi Tri Hita Karana (keharmonisan dengan Tuhan, sesama, dan alam). Subak telah 
                diakui UNESCO sebagai Warisan Budaya Dunia.
              </p>
              <div className="mt-3 p-3 bg-amber-50 rounded-lg">
                <p className="text-xs font-medium mb-1">Prinsip Utama:</p>
                <ul className="text-xs space-y-1 list-disc list-inside text-muted-foreground">
                  <li>Pembagian air secara demokratis dan adil</li>
                  <li>Ritual keagamaan untuk keseimbangan alam</li>
                  <li>Gotong royong dalam pengelolaan sawah</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <Separator />

        {/* Tumpang Sari */}
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <Badge variant="secondary" className="mt-1">Nusantara</Badge>
            <div className="flex-1">
              <h3 className="font-semibold mb-2">Tumpang Sari</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Sistem pertanian tradisional yang menanam berbagai jenis tanaman dalam satu lahan secara 
                bersamaan. Metode ini memaksimalkan penggunaan lahan, mengurangi hama, dan meningkatkan 
                kesuburan tanah secara alami.
              </p>
              <div className="mt-3 grid grid-cols-2 gap-2">
                <div className="text-xs p-2 bg-purple-50 rounded">
                  <span className="font-medium">Manfaat:</span> Efisiensi lahan
                </div>
                <div className="text-xs p-2 bg-pink-50 rounded">
                  <span className="font-medium">Keuntungan:</span> Diversifikasi hasil
                </div>
              </div>
            </div>
          </div>
        </div>

        <Separator />

        {/* Sasi */}
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <Badge variant="secondary" className="mt-1">Maluku</Badge>
            <div className="flex-1">
              <h3 className="font-semibold mb-2">Sasi</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Larangan adat untuk mengeksploitasi sumber daya alam pada waktu tertentu agar dapat 
                pulih dan berkembang. Sistem ini mengajarkan konservasi dan keberlanjutan sumber daya alam.
              </p>
            </div>
          </div>
        </div>

        <Separator />

        {/* Jajar Legowo */}
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <Badge variant="secondary" className="mt-1">Modern</Badge>
            <div className="flex-1">
              <h3 className="font-semibold mb-2">Jajar Legowo</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Adaptasi modern dari sistem tanam tradisional dengan pola barisan yang diperlebar. 
                Meningkatkan produktivitas hingga 20% karena sirkulasi udara dan cahaya lebih baik.
              </p>
              <div className="mt-3 p-3 bg-green-50 rounded-lg">
                <p className="text-xs font-medium mb-1">Keunggulan:</p>
                <ul className="text-xs space-y-1 list-disc list-inside text-muted-foreground">
                  <li>Peningkatan produktivitas signifikan</li>
                  <li>Kemudahan dalam perawatan tanaman</li>
                  <li>Populasi tanaman per hektar lebih optimal</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}