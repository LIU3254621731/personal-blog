"use client";

import { useState, useEffect, useCallback } from "react";

/**
 * Hook: check if current user is authenticated.
 * Returns { authed, loading, checkAuth }.
 * Used by inline admin controls throughout the site.
 */
export function useAuth() {
  const [authed, setAuthed] = useState(false);
  const [loading, setLoading] = useState(true);

  const checkAuth = useCallback(async () => {
    try {
      const res = await fetch("/api/auth/check");
      const data = await res.json();
      setAuthed(data.authenticated === true);
    } catch {
      setAuthed(false);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  // Listen for login/logout events
  useEffect(() => {
    function onFocus() { checkAuth(); }
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [checkAuth]);

  return { authed, loading, checkAuth };
}
