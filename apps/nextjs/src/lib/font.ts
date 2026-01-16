import { Inter } from "next/font/google";
import { GeistMono } from "geist/font/mono";

export const fontSans = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
  variable: "--font-sans",
  display: "swap",
  adjustFontFallback: true,
  fallback: ["system-ui", "arial"],
});

// Tetap pakai Geist Mono untuk monospace
export const fontMono = GeistMono;
