import { Montserrat } from "next/font/google";
import { GeistMono } from "geist/font/mono";

export const fontMonserrat = Montserrat({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
  variable: "--font-sans",
  display: "swap",
});

// Tetap pakai Geist Mono untuk monospace
export const fontMono = GeistMono;
