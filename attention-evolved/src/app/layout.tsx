import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ThemeScript } from "@/components/ThemeScript";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Attention, Evolved | A Visual History of Attention Mechanisms",
  description: "An interactive chronological exploration of how attention mechanisms evolved from 2017 to today. Each technique is a response to a bottleneck—compute, memory, context length, or bandwidth.",
  keywords: ["attention mechanism", "transformer", "RoPE", "MQA", "GQA", "linear attention", "deep learning", "machine learning"],
  authors: [{ name: "Tushar Gupta" }],
  openGraph: {
    title: "Attention, Evolved",
    description: "A visual history of attention mechanisms in deep learning",
    type: "website",
  },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        <ThemeScript />
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
          integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV"
          crossOrigin="anonymous"
        />
      </head>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
