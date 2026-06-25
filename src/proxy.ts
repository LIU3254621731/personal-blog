import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// ─── Rate Limiter (in-memory, per IP) ─────────────────

interface RateEntry { count: number; resetAt: number; }
const rateMap = new Map<string, RateEntry>();

setInterval(() => {
  const now = Date.now();
  for (const [k, v] of rateMap) { if (now > v.resetAt) rateMap.delete(k); }
}, 60_000);

function getClientIp(request: NextRequest): string {
  return (
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    request.headers.get("x-real-ip") ||
    "127.0.0.1"
  );
}

function checkRate(request: NextRequest, maxRequests: number, windowMs: number): NextResponse | null {
  if (!request.nextUrl.pathname.startsWith("/api/")) return null;
  const ip = getClientIp(request);
  const key = `${ip}:${request.nextUrl.pathname}`;
  const now = Date.now();
  let entry = rateMap.get(key);
  if (!entry || now > entry.resetAt) { entry = { count: 0, resetAt: now + windowMs }; rateMap.set(key, entry); }
  const isAuth = request.nextUrl.pathname.startsWith("/api/auth");
  const limit = isAuth ? Math.min(maxRequests, 5) : maxRequests;
  entry.count++;
  if (entry.count > limit) {
    return NextResponse.json({ error: "请求过于频繁，请稍后再试" }, { status: 429 });
  }
  return null;
}

// ─── Security Headers ──────────────────────────────────

function addSecurityHeaders(response: NextResponse): void {
  response.headers.set("X-Content-Type-Options", "nosniff");
  response.headers.set("X-Frame-Options", "DENY");
  response.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  response.headers.set("Permissions-Policy", "camera=(), microphone=(), geolocation=(), interest-cohort=()");
  if (process.env.NODE_ENV === "production") {
    response.headers.set("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload");
  }
}

// ─── Proxy ─────────────────────────────────────────────

export function proxy(request: NextRequest) {
  // 1. Rate limiting
  const rateResponse = checkRate(request, 60, 60_000);
  if (rateResponse) { addSecurityHeaders(rateResponse); return rateResponse; }

  // 2. CSRF check for state-changing API requests
  if (
    request.nextUrl.pathname.startsWith("/api/") &&
    !["GET", "HEAD", "OPTIONS"].includes(request.method) &&
    !request.nextUrl.pathname.startsWith("/api/auth/login") &&
    !request.nextUrl.pathname.startsWith("/api/auth/logout")
  ) {
    const csrfCookie = request.cookies.get("csrf_token");
    const csrfHeader = request.headers.get("x-csrf-token");
    if (!csrfCookie?.value || csrfCookie.value !== csrfHeader) {
      return NextResponse.json({ error: "CSRF 校验失败" }, { status: 403 });
    }
  }

  const response = NextResponse.next();
  response.headers.set("x-pathname", request.nextUrl.pathname);
  addSecurityHeaders(response);
  return response;
}
