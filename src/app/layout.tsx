import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Noto_Sans_SC } from "next/font/google";
import { GeistSans } from "geist/font/sans";
import { Navigation } from "@/components/layout/navigation";
import { Footer } from "@/components/layout/footer";
import { BackToTop } from "@/components/layout/back-to-top";
import { CommandPalette } from "@/components/search/command-palette";
import { siteConfig } from "@/data/site";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

const notoSansSC = Noto_Sans_SC({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-cjk",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: siteConfig.name + " — " + siteConfig.title,
    template: "%s — " + siteConfig.name,
  },
  description: siteConfig.description,
  keywords: [
    "AI",
    "Machine Learning",
    "Deep Learning",
    "Personal Blog",
    "Portfolio",
    "Computer Vision",
    "Full Stack",
  ],
  authors: [{ name: siteConfig.author }],
  creator: siteConfig.author,
  openGraph: {
    title: siteConfig.name + " — " + siteConfig.title,
    description: siteConfig.description,
    type: "website",
    locale: "zh_CN",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default async function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const fontClasses = [
    inter.variable,
    GeistSans.variable,
    jetbrains.variable,
    notoSansSC.variable,
    "h-full",
    "antialiased",
  ].join(" ");

  return (
    <html lang="zh-CN" className={fontClasses} suppressHydrationWarning>
      <head>
        <link
          rel="icon"
          href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>✦</text></svg>"
        />
        <script
          dangerouslySetInnerHTML={{
            __html:
              "try{var t=localStorage.getItem('theme');if(t==='dark'||(!t&&matchMedia('(prefers-color-scheme:dark)').matches))document.documentElement.classList.add('dark')}catch(e){}",
          }}
        />
      </head>
      <body className="min-h-full flex flex-col font-body relative">
        <div className="ambient-blob w-[700px] h-[700px] bg-[#818CF8] -top-60 -right-60" />
        <div className="ambient-blob w-[500px] h-[500px] bg-[#4F46E5] bottom-0 -left-40" />
        <Navigation />
        <main className="flex-1 pt-28">{children}</main>
        <Footer />
        <BackToTop />
        <CommandPalette />
      </body>
    </html>
  );
}
