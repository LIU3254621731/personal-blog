import type { Metadata } from "next";
import { Inter, Playfair_Display, JetBrains_Mono } from "next/font/google";
import { Navigation } from "@/components/layout/navigation";
import { Footer } from "@/components/layout/footer";
import { BackToTop } from "@/components/layout/back-to-top";
import { siteConfig } from "@/data/site";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-body", display: "swap" });
const playfair = Playfair_Display({ subsets: ["latin"], variable: "--font-display", display: "swap", weight: ["400", "500", "600", "700"] });
const jetbrains = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: siteConfig.author + " — " + siteConfig.title,
  description: siteConfig.description,
};

export default async function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const fontClasses = inter.variable + " " + playfair.variable + " " + jetbrains.variable + " h-full antialiased";

  return (
    <html lang="zh-CN" className={fontClasses} suppressHydrationWarning>
      <head>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>✦</text></svg>" />
        <script dangerouslySetInnerHTML={{ __html: "try{var t=localStorage.getItem('theme');if(t==='dark'||(!t&&matchMedia('(prefers-color-scheme:dark)').matches))document.documentElement.classList.add('dark')}catch(e){}" }} />
      </head>
      <body className="min-h-full flex flex-col font-body relative">
        <div className="ambient-blob w-[600px] h-[600px] bg-[#d4c5b2] dark:bg-[#3a3040] -top-40 -right-40" />
        <div className="ambient-blob w-[500px] h-[500px] bg-[#c5c9c0] dark:bg-[#2a3040] bottom-0 -left-40" />
        <Navigation />
        <main className="flex-1 pt-28">{children}</main>
        <Footer />
        <BackToTop />
      </body>
    </html>
  );
}
