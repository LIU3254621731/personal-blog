"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Save, Check, Settings, Key, User, Globe } from "lucide-react";

export default function SettingsPage() {
  const [config, setConfig] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"site" | "profile" | "password">("site");

  useEffect(() => {
    fetch("/api/site-config")
      .then((r) => r.json())
      .then(setConfig)
      .finally(() => setLoading(false));
  }, []);

  async function handleSave() {
    await fetch("/api/site-config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  }

  function update(key: string, value: string) {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }

  if (loading) return <div className="text-center text-text-tertiary py-32">加载中...</div>;

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="flex items-center gap-3 mb-8">
          <div className="w-9 h-9 rounded-xl bg-accent-light dark:bg-accent-light/20 flex items-center justify-center">
            <Settings size={18} className="text-accent" />
          </div>
          <h1 className="font-display text-2xl font-semibold text-text-primary">个人设置</h1>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-8 p-1 glass rounded-xl">
          {[
            { id: "site" as const, label: "站点", icon: Globe },
            { id: "profile" as const, label: "个人信息", icon: User },
            { id: "password" as const, label: "密码", icon: Key },
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                tab === t.id
                  ? "bg-bg-primary text-text-primary shadow-sm"
                  : "text-text-tertiary hover:text-text-secondary"
              }`}
            >
              <t.icon size={15} />
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab: Site */}
        {tab === "site" && (
          <div className="glass rounded-2xl p-6 md:p-8 space-y-5">
            <Field label="站点标题" value={config.title || ""} onChange={(v) => update("title", v)} placeholder="Wenlin Lab" />
            <Field label="副标题" value={config.subtitle || ""} onChange={(v) => update("subtitle", v)} placeholder="Building, Learning, Sharing" />
            <Field label="站点描述" value={config.description || ""} onChange={(v) => update("description", v)} placeholder="个人数字实验室" type="textarea" />
            <Field label="作者名" value={config.author || ""} onChange={(v) => update("author", v)} placeholder="夏阳" />
            <Field label="GitHub" value={config.github || ""} onChange={(v) => update("github", v)} placeholder="https://github.com/..." />
            <div className="flex items-center gap-3 pt-2">
              <button onClick={handleSave} className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-all">
                <Save size={16} /> 保存
              </button>
              {saved && <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-1 text-sm text-green-600"><Check size={14} /> 已保存</motion.span>}
            </div>
          </div>
        )}

        {/* Tab: Profile */}
        {tab === "profile" && (
          <div className="glass rounded-2xl p-6 md:p-8 space-y-5">
            <Field label="姓名" value={config.name || ""} onChange={(v) => update("name", v)} placeholder="夏阳" />
            <Field label="一句话介绍" value={config.tagline || ""} onChange={(v) => update("tagline", v)} placeholder="AI Developer · Student · Builder" />
            <Field label="个人简介" value={config.bio || ""} onChange={(v) => update("bio", v)} placeholder="关于你的简介..." type="textarea" />
            <Field label="QQ 邮箱" value={config.email || ""} onChange={(v) => update("email", v)} placeholder="3254621731@qq.com" />
            <Field label="Gmail" value={config.emailGmail || ""} onChange={(v) => update("emailGmail", v)} placeholder="l18398916038@gmail.com" />
            <div className="flex items-center gap-3 pt-2">
              <button onClick={handleSave} className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-all">
                <Save size={16} /> 保存
              </button>
            </div>
          </div>
        )}

        {/* Tab: Password */}
        {tab === "password" && (
          <PasswordSection />
        )}
      </motion.div>
    </div>
  );
}

/* ─── Password Section ───────────────────────────────── */

function PasswordSection() {
  const [cp, setCp] = useState("");
  const [np, setNp] = useState("");
  const [npc, setNpc] = useState("");
  const [msg, setMsg] = useState<{ t: string; ok: boolean } | null>(null);
  const [l, setL] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault(); setMsg(null);
    if (np !== npc) { setMsg({ t: "两次输入不一致", ok: false }); return; }
    setL(true);
    try {
      const res = await fetch("/api/auth/password", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ currentPassword: cp, newPassword: np }),
      });
      const d = await res.json();
      setMsg({ t: res.ok ? "密码修改成功" : (d.error || "失败"), ok: res.ok });
      if (res.ok) { setCp(""); setNp(""); setNpc(""); }
    } catch { setMsg({ t: "网络错误", ok: false }); }
    setL(false);
  }

  const ic = "w-full px-4 py-3 rounded-xl border border-border-medium bg-bg-primary/50 text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all";
  return (
    <form onSubmit={handleSubmit} className="glass rounded-2xl p-6 md:p-8 space-y-4 max-w-sm">
      <h2 className="font-display text-lg font-semibold text-text-primary mb-2">修改登录密码</h2>
      <input type="password" value={cp} onChange={(e) => setCp(e.target.value)} placeholder="当前密码" className={ic} required />
      <input type="password" value={np} onChange={(e) => setNp(e.target.value)} placeholder="新密码（至少 6 位）" className={ic} required minLength={6} />
      <input type="password" value={npc} onChange={(e) => setNpc(e.target.value)} placeholder="确认新密码" className={ic} required />
      {msg && <p className={`text-sm ${msg.ok ? "text-green-600" : "text-red-500"}`}>{msg.t}</p>}
      <button type="submit" disabled={l} className="px-5 py-2.5 rounded-xl bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-all disabled:opacity-50">
        {l ? "修改中..." : "修改密码"}
      </button>
    </form>
  );
}

/* ─── Field ───────────────────────────────────────────── */

function Field({ label, value, onChange, placeholder, type = "text" }: {
  label: string; value: string; onChange: (v: string) => void; placeholder: string; type?: string;
}) {
  const ic = "w-full px-4 py-3 rounded-xl border border-border-medium bg-bg-primary/50 text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all";
  return (
    <div>
      <label className="block text-sm font-medium text-text-secondary mb-2">{label}</label>
      {type === "textarea" ? (
        <textarea value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} rows={3} className={`${ic} resize-y`} />
      ) : (
        <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className={ic} />
      )}
    </div>
  );
}
