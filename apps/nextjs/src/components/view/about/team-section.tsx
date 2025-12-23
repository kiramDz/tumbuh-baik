"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Users } from "lucide-react";

export default function TeamSection() {
  const teamMembers = [
    {
      name: "Dr. Budi Santoso",
      role: "Founder & CEO",
      description: "Ahli pertanian dengan pengalaman 20 tahun",
      avatar: "BS"
    },
    {
      name: "Siti Nurhaliza",
      role: "Head of Technology",
      description: "Spesialis sistem informasi pertanian",
      avatar: "SN"
    },
    {
      name: "Ahmad Fauzi",
      role: "Field Coordinator",
      description: "Menghubungkan petani dengan teknologi",
      avatar: "AF"
    }
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Users className="h-5 w-5 text-blue-600" />
          Tim Kami
        </CardTitle>
        <CardDescription>
          Orang-orang di balik Tumbuh Baik
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid md:grid-cols-3 gap-6">
          {teamMembers.map((member, index) => (
            <Card key={index} className="text-center">
              <CardContent className="pt-6">
                <div className="h-20 w-20 rounded-full bg-gradient-to-br from-green-400 to-blue-500 flex items-center justify-center text-white font-bold text-xl mx-auto mb-4">
                  {member.avatar}
                </div>
                <h3 className="font-semibold mb-1">{member.name}</h3>
                <p className="text-sm text-muted-foreground mb-2">{member.role}</p>
                <p className="text-xs text-muted-foreground">{member.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}