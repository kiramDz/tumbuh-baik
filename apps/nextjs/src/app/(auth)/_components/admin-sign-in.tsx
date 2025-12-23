"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Metadata } from "next";
import Link from "next/link";
import { Loader2, Eye, EyeOff, Mail, Lock, ArrowRight } from "lucide-react";

import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { signIn } from "@/lib/better-auth/auth-client";

export const metadata: Metadata = {
  title: "Admin Sign In - Authentication",
  description: "Admin authentication portal for Tumbuh Baik platform.",
};

export default function AdminSignIn() {
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();

  return (
    <div className="relative min-h-screen flex-col items-center justify-center md:grid lg:max-w-none lg:grid-cols-2 lg:px-0 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      
      {/* Left Panel */}
      <div className="relative hidden h-full flex-col p-10 text-white lg:flex dark:border-r overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-gray-800 via-gray-900 to-black" />
        <div className="absolute inset-0 bg-black/20" />

        {/* Decorative elements */}
        <div className="absolute top-20 right-20 w-32 h-32 bg-white/5 rounded-full blur-xl" />
        <div className="absolute bottom-32 left-16 w-24 h-24 bg-white/5 rounded-full blur-lg" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gray-700/10 rounded-full blur-3xl" />

        <div className="relative z-20 flex items-center text-xl font-bold text-white">
          <div className="mr-3 p-2 bg-white/20 rounded-lg backdrop-blur-sm">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-6 w-6 text-white"
            >
              <path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3" />
            </svg>
          </div>
          ZonaPETIK
        </div>

        <div className="relative z-20 mt-auto">
          <Card className="bg-white/10 backdrop-blur-sm border-white/20">
            <CardContent className="p-8">
              <blockquote className="space-y-4">
                <p className="text-lg leading-relaxed font-medium text-white">
                  &ldquo;Climate change is real. It is happening right now, it is the most urgent threat facing our entire species and we need to work collectively together and stop procrastinating.&rdquo;
                </p>
                <footer className="text-sm opacity-90 font-medium text-white">— Leonardo Di Caprio</footer>
              </blockquote>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Right Panel */}
      <div className="flex h-full items-center justify-center p-4 lg:p-8">
        <div className="flex w-full max-w-md flex-col items-center justify-center space-y-6">
          
          {/* Mobile Header */}
          <div className="text-center space-y-2 lg:hidden">
            <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-200">Tumbuh Baik</h1>
            <p className="text-sm text-gray-600 dark:text-gray-400">Admin Portal</p>
          </div>

          {/* Sign In Card */}
          <Card className="w-full shadow-lg border">
            <CardHeader className="space-y-3">
              <div className="flex flex-col items-center space-y-3">
                <div className="p-3 bg-gray-100 dark:bg-gray-800 rounded-full">
                  <Lock className="h-6 w-6 text-gray-700 dark:text-gray-300" />
                </div>
                <div className="text-center space-y-1">
                  <CardTitle className="text-2xl font-bold">
                    Sign In
                  </CardTitle>
                  <CardDescription className="text-sm">
                    Enter your credentials to access the admin dashboard
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <Separator />

            <CardContent className="pt-6 space-y-4">
              
              {/* Error Alert */}
              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {/* Email Field */}
              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="admin@tumbuhbaik.com"
                    required
                    className="pl-10"
                    onChange={(e) => setEmail(e.target.value)}
                    value={email}
                    disabled={loading}
                  />
                </div>
              </div>

              {/* Password Field */}
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter your password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-10 pr-12"
                    disabled={loading}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !loading) {
                        // Trigger sign in on Enter key
                        document.getElementById("signin-button")?.click();
                      }
                    }}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                    disabled={loading}
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <Eye className="h-4 w-4 text-muted-foreground" />
                    )}
                  </Button>
                </div>
              </div>

              {/* Sign In Button */}
              <Button
                id="signin-button"
                type="submit"
                disabled={loading || !email || !password}
                className="w-full"
                size="lg"
                onClick={async () => {
                  try {
                    setLoading(true);
                    setError("");
                    const result = await signIn.email({ email, password });
                    console.log("SIGNIN RESULT:", result);

                    if (result?.error) {
                      console.error("SIGNIN ERROR:", result.error);
                      setError(result.error.message || "Login failed");
                    } else {
                      router.push("/dashboard");
                    }
                  } catch (err) {
                    console.error("SIGNIN CATCH ERROR:", err);
                    setError("Unexpected error occurred");
                  } finally {
                    setLoading(false);
                  }
                }}
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Signing In...
                  </>
                ) : (
                  <>
                    Sign In
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </>
                )}
              </Button>

            </CardContent>

            <Separator />

            <CardFooter className="flex flex-col space-y-4 pt-6">
              <div className="text-sm text-center text-muted-foreground">
                Don&apos;t have an account?{" "}
                <Link href="/sign-up" className="font-medium text-primary underline underline-offset-4 hover:text-primary/80 transition-colors">
                  Sign up now
                </Link>
              </div>
            </CardFooter>
          </Card>

          {/* Footer Text */}
          <p className="text-xs text-center text-muted-foreground max-w-sm px-4">
            By signing in, you agree to our{" "}
            <Link href="/terms" className="underline underline-offset-4 hover:text-primary">
              Terms of Service
            </Link>{" "}
            and{" "}
            <Link href="/privacy" className="underline underline-offset-4 hover:text-primary">
              Privacy Policy
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
