"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Loader2, Eye, EyeOff, Mail, Lock, User, ArrowRight, UserPlus } from "lucide-react";
import { toast } from "sonner";

import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { signUp } from "@/lib/better-auth/auth-client";

export function AdminSignUp() {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirmation, setPasswordConfirmation] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();

  const handleSignUp = async () => {
    // Validation
    if (!firstName || !lastName || !email || !password || !passwordConfirmation) {
      setError("All fields are required");
      return;
    }

    if (password !== passwordConfirmation) {
      setError("Passwords do not match");
      return;
    }

    if (password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }

    setError("");

    await signUp.email({
      email,
      password,
      name: `${firstName} ${lastName}`,
      callbackURL: "/dashboard",
      fetchOptions: {
        onRequest: () => {
          setLoading(true);
        },
        onResponse: () => {
          setLoading(false);
        },
        onError: (ctx) => {
          setError(ctx.error.message);
          toast.error(ctx.error.message);
        },
        onSuccess: async () => {
          toast.success("Account created successfully!");
          router.push("/dashboard");
        },
      },
    });
  };

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
                  &ldquo;Join us in making agriculture smarter and more sustainable through technology and data-driven solutions.&rdquo;
                </p>
                <footer className="text-sm opacity-90 font-medium text-white">— Tumbuh Baik Team</footer>
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
            <p className="text-sm text-gray-600 dark:text-gray-400">Create Your Account</p>
          </div>

          {/* Sign Up Card */}
          <Card className="w-full shadow-lg border">
            <CardHeader className="space-y-3">
              <div className="flex flex-col items-center space-y-3">
                <div className="p-3 bg-gray-100 dark:bg-gray-800 rounded-full">
                  <UserPlus className="h-6 w-6 text-gray-700 dark:text-gray-300" />
                </div>
                <div className="text-center space-y-1">
                  <CardTitle className="text-2xl font-bold">
                    Create Account
                  </CardTitle>
                  <CardDescription className="text-sm">
                    Enter your information to create an account
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

              {/* Name Fields */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="first-name">First Name</Label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="first-name"
                      placeholder="John"
                      required
                      className="pl-10"
                      onChange={(e) => setFirstName(e.target.value)}
                      value={firstName}
                      disabled={loading}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="last-name">Last Name</Label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="last-name"
                      placeholder="Doe"
                      required
                      className="pl-10"
                      onChange={(e) => setLastName(e.target.value)}
                      value={lastName}
                      disabled={loading}
                    />
                  </div>
                </div>
              </div>

              {/* Email Field */}
              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="john.doe@example.com"
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
                    placeholder="Create a strong password"
                    autoComplete="new-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-10 pr-12"
                    disabled={loading}
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
                <p className="text-xs text-muted-foreground">
                  Must be at least 8 characters
                </p>
              </div>

              {/* Confirm Password Field */}
              <div className="space-y-2">
                <Label htmlFor="password-confirmation">Confirm Password</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="password-confirmation"
                    type={showConfirmPassword ? "text" : "password"}
                    placeholder="Confirm your password"
                    autoComplete="new-password"
                    value={passwordConfirmation}
                    onChange={(e) => setPasswordConfirmation(e.target.value)}
                    className="pl-10 pr-12"
                    disabled={loading}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !loading) {
                        handleSignUp();
                      }
                    }}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                    disabled={loading}
                  >
                    {showConfirmPassword ? (
                      <EyeOff className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <Eye className="h-4 w-4 text-muted-foreground" />
                    )}
                  </Button>
                </div>
              </div>

              {/* Sign Up Button */}
              <Button
                type="submit"
                disabled={loading || !firstName || !lastName || !email || !password || !passwordConfirmation}
                className="w-full"
                size="lg"
                onClick={handleSignUp}
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Creating Account...
                  </>
                ) : (
                  <>
                    Create Account
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </>
                )}
              </Button>

            </CardContent>

            <Separator />

            <CardFooter className="flex flex-col space-y-4 pt-6">
              <div className="text-sm text-center text-muted-foreground">
                Already have an account?{" "}
                <Link href="/sign-in" className="font-medium text-primary underline underline-offset-4 hover:text-primary/80 transition-colors">
                  Sign in
                </Link>
              </div>
            </CardFooter>
          </Card>
        </div>
      </div>
    </div>
  );
}
