"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Metadata } from "next";
import Link from "next/link";
import { Loader2 } from "lucide-react";

import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { signIn } from "@/lib/better-auth/auth-client";

export const metadata: Metadata = {
  title: "Authentication",
  description: "Authentication forms built using the components.",
};

export default function AdminSignIn() {
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();

  return (
    <div className="relative h-screen flex-col items-center justify-center md:grid lg:max-w-none lg:grid-cols-2 lg:px-0">
      <Link href="/sign-in" className={cn(buttonVariants({ variant: "ghost" }), "absolute top-4 right-4 hidden md:top-8 md:right-8")}>
        Sign In
      </Link>
      <div className="bg-muted relative hidden h-full flex-col p-10 text-black lg:flex dark:border-r">
        <div className="absolute inset-0 bg-[#d0e8ff]" />
        <div className="relative z-20 flex items-center text-lg font-medium">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2 h-6 w-6">
            <path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3" />
          </svg>
          Tumbuh Baik
        </div>
        <div className="relative z-20 mt-auto">
          <blockquote className="space-y-2 text-black">
            <p className="text-lg">&ldquo;Climate change is real. It is happening right now, it is the most urgent threat facing our entire species and we need to work collectively together and stop procrastinating..&rdquo;</p>
            <footer className="text-sm">Leonardo Di Caprio</footer>
          </blockquote>
        </div>
      </div>
      <div className="flex h-full items-center justify-center p-4 lg:p-8">
        <div className="flex w-full max-w-md flex-col items-center justify-center space-y-6">
          <Card className="gap-4">
            <CardHeader>
              <CardTitle className="text-lg md:text-xl">Sign In</CardTitle>
              <CardDescription className="text-xs md:text-sm">Enter your email below to login to your account</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="m@example.com"
                    required
                    onChange={(e) => {
                      setEmail(e.target.value);
                    }}
                    value={email}
                  />
                </div>

                <div className="grid gap-2">
                  <div className="flex items-center">
                    <Label htmlFor="password">Password</Label>
                  </div>
                  <div className="relative rounded-md">
                    <Input id="password" type="password" placeholder="password" autoComplete="password" value={password} onChange={(e) => setPassword(e.target.value)} />
                  </div>
                </div>

                <Button
                  type="submit"
                  className="w-full"
                  disabled={loading}
                  onClick={async () => {
                    try {
                      setLoading(true);
                      const result = await signIn.email({ email, password });
                      console.log("SIGNIN RESULT:", result);

                      if (result?.error) {
                        console.error("SIGNIN ERROR:", result.error);
                        // toast.error(result.error.message || "Login failed");
                      } else {
                        router.push("/dashboard");
                      }
                    } catch (err) {
                      console.error("SIGNIN CATCH ERROR:", err);
                      // toast.error("Unexpected error occurred");
                    } finally {
                      setLoading(false);
                    }
                  }}
                >
                  {loading ? <Loader2 size={16} className="animate-spin" /> : "Login"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
