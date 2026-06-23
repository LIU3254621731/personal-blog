"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    if (res.ok) {
      router.push("/admin");
    } else {
      setError("密码错误");
    }
    setLoading(false);
  }

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh", background: "#f5f4f1" }}>
      <form onSubmit={handleSubmit} style={{ background: "#fff", padding: 40, borderRadius: 16, width: 360, boxShadow: "0 4px 24px rgba(0,0,0,0.06)" }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 8 }}>CMS 登录</h1>
        <p style={{ color: "#999", fontSize: 14, marginBottom: 24 }}>请输入管理密码</p>
        <input
          type="password" value={password} onChange={(e) => setPassword(e.target.value)}
          placeholder="密码"
          style={{ width: "100%", padding: "12px 16px", borderRadius: 10, border: "1px solid #e0ded9", fontSize: 15, outline: "none", boxSizing: "border-box", marginBottom: 16 }}
          autoFocus
        />
        {error && <p style={{ color: "#e53e3e", fontSize: 13, marginBottom: 12 }}>{error}</p>}
        <button type="submit" disabled={loading}
          style={{ width: "100%", padding: "12px", borderRadius: 10, border: "none", background: "#1c1c1e", color: "#fff", fontSize: 15, fontWeight: 600, cursor: "pointer" }}
        >
          {loading ? "登录中..." : "登录"}
        </button>
      </form>
    </div>
  );
}
