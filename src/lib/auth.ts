/**
 * Authentication module — bcrypt + JWT (jose).
 *
 * Configuration:
 *   auth.config.json  — { adminPasswordHash, jwtSecret }
 *   .env             — JWT_SECRET (fallback)
 *
 * Generate credentials:  node scripts/setup-env.js
 */

import { cookies } from "next/headers";
import { readFileSync, writeFileSync } from "fs";
import path from "path";
import bcrypt from "bcryptjs";
import { SignJWT, jwtVerify } from "jose";
import crypto from "crypto";

const COOKIE_NAME = "admin_token";
const CSRF_COOKIE = "csrf_token";
const TOKEN_MAX_AGE = 60 * 60 * 24 * 7; // 7 days (seconds)

interface AuthConfig {
  adminPasswordHash: string;
  jwtSecret: string;
}

let _config: AuthConfig | null = null;

function getConfig(): AuthConfig {
  if (_config) return _config;

  // Prefer auth.config.json (avoids dotenv-expand issues with $ in bcrypt hashes)
  try {
    const configPath = path.join(process.cwd(), "auth.config.json");
    const raw = readFileSync(configPath, "utf8");
    _config = JSON.parse(raw) as AuthConfig;
    if (_config.adminPasswordHash && _config.jwtSecret) return _config;
  } catch {
    // Fall through to env
  }

  // Fallback: read from environment variables
  _config = {
    adminPasswordHash: process.env.ADMIN_PASSWORD_HASH || "",
    jwtSecret: process.env.JWT_SECRET || "",
  };

  return _config;
}

function getSecret(): Uint8Array {
  const secret = getConfig().jwtSecret;
  if (!secret) throw new Error("jwtSecret is not set — run: node scripts/setup-env.js");
  return new TextEncoder().encode(secret);
}

function getPasswordHash(): string {
  const hash = getConfig().adminPasswordHash;
  if (!hash) throw new Error("adminPasswordHash is not set — run: node scripts/setup-env.js");
  return hash;
}

/** Verify plain-text password against the stored bcrypt hash. */
export async function verifyPassword(password: string): Promise<boolean> {
  try {
    return await bcrypt.compare(password, getPasswordHash());
  } catch {
    return false;
  }
}

/** Sign a JWT and set it as an httpOnly cookie. Returns true on success. */
export async function login(password: string): Promise<boolean> {
  const valid = await verifyPassword(password);
  if (!valid) return false;

  const secret = getSecret();
  const token = await new SignJWT({ sub: "admin" })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(`${TOKEN_MAX_AGE}s`)
    .sign(secret);

  const cookieStore = await cookies();
  cookieStore.set(COOKIE_NAME, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: TOKEN_MAX_AGE,
  });

  // Set CSRF token (JS-readable, for double-submit pattern)
  const csrfToken = crypto.randomUUID();
  cookieStore.set(CSRF_COOKIE, csrfToken, {
    httpOnly: false, // JS must read it
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: TOKEN_MAX_AGE,
  });

  return true;
}

/** Delete the auth cookie and CSRF token. */
export async function logout(): Promise<void> {
  const cookieStore = await cookies();
  cookieStore.delete(COOKIE_NAME);
  cookieStore.delete(CSRF_COOKIE);
}

/** Verify the JWT from the cookie. Returns true if valid. */
export async function isAuthenticated(): Promise<boolean> {
  try {
    const cookieStore = await cookies();
    const token = cookieStore.get(COOKIE_NAME);
    if (!token?.value) return false;

    const secret = getSecret();
    await jwtVerify(token.value, secret);
    return true;
  } catch {
    return false;
  }
}

/** Change the admin password. Writes new bcrypt hash to auth.config.json. */
export async function changePassword(
  currentPassword: string,
  newPassword: string,
): Promise<{ success: boolean; error?: string }> {
  if (newPassword.length < 6) {
    return { success: false, error: "新密码至少需要 6 个字符" };
  }

  const valid = await verifyPassword(currentPassword);
  if (!valid) {
    return { success: false, error: "当前密码错误" };
  }

  const newHash = await bcrypt.hash(newPassword, 12);
  const configPath = path.join(process.cwd(), "auth.config.json");
  const config = JSON.parse(readFileSync(configPath, "utf8"));
  config.adminPasswordHash = newHash;

  writeFileSync(configPath, JSON.stringify(config, null, 2), "utf8");

  // Bust the in-memory cache
  _config = null;

  return { success: true };
}

/** Verify CSRF double-submit cookie against request header. */
export async function verifyCsrf(request: Request): Promise<boolean> {
  // Skip CSRF check for GET/HEAD/OPTIONS
  const method = (request as { method?: string }).method || "GET";
  if (["GET", "HEAD", "OPTIONS"].includes(method.toUpperCase())) return true;

  try {
    const cookieStore = await cookies();
    const csrfCookie = cookieStore.get(CSRF_COOKIE);
    if (!csrfCookie?.value) return false;

    const csrfHeader = (request as Request).headers.get("x-csrf-token");
    return csrfCookie.value === csrfHeader;
  } catch {
    return false;
  }
}
