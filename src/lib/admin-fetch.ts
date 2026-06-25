/**
 * Fetch wrapper for admin pages that automatically includes the CSRF token.
 *
 * Usage (in any admin client component):
 *   import { adminFetch } from "@/lib/admin-fetch";
 *   const res = await adminFetch("/api/posts", { method: "POST", body: JSON.stringify(data) });
 */

function getCsrfToken(): string {
  if (typeof document === "undefined") return "";
  const match = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/);
  return match ? match[1] : "";
}

export async function adminFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");

  // Attach CSRF token for state-changing methods
  const method = (init?.method || "GET").toUpperCase();
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    headers.set("x-csrf-token", getCsrfToken());
  }

  return fetch(input, { ...init, headers });
}
