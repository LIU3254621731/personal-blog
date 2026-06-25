import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // EdgeOne needs standalone output for SSR
  output: "standalone",

  // Keep SQLite external for local dev; EdgeOne won't have it
  serverExternalPackages: ["better-sqlite3"],

  // Disable image optimization (EdgeOne has its own)
  images: {
    unoptimized: true,
  },

  // Security headers
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
              "style-src 'self' 'unsafe-inline'",
              "img-src 'self' data: blob: https:",
              "font-src 'self' data:",
              "connect-src 'self' https:",
              "frame-src 'none'",
              "object-src 'none'",
              "base-uri 'self'",
              "form-action 'self'",
            ].join("; "),
          },
        ],
      },
    ];
  },
};

export default nextConfig;
