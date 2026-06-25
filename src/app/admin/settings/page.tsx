"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Save, Check } from "lucide-react";

export default function SettingsPage() {
  const [config, setConfig] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(true);

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

  if (loading) {
    return (
      <div className="text-center text-text-tertiary py-20">加载中...</div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="flex items-center justify-between mb-8">
        <h1 className="font-display text-2xl font-semibold text-text-primary">
          站点设置
        </h1>
        <div className="flex items-center gap-3">
          {saved && (
            <motion.span
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-1.5 text-sm text-green-600 dark:text-green-400"
            >
              <Check size={16} />
              已保存
            </motion.span>
          )}
          <button
            onClick={handleSave}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent text-white text-sm font-medium
              hover:bg-accent/90 active:scale-[0.98] transition-all duration-200"
          >
            <Save size={17} />
            保存设置
          </button>
        </div>
      </div>

      {/* Site Config */}
      <div className="glass rounded-2xl p-6 md:p-8 space-y-6">
        <Field
          label="站点标题"
          value={config.title || ""}
          onChange={(v) => update("title", v)}
          placeholder="Wenlin Lab"
        />
        <Field
          label="副标题"
          value={config.subtitle || ""}
          onChange={(v) => update("subtitle", v)}
          placeholder="Building, Learning, Sharing"
        />
        <Field
          label="站点描述"
          value={config.description || ""}
          onChange={(v) => update("description", v)}
          placeholder="记录关于代码、AI 和探索的个人空间。"
          type="textarea"
        />
        <Field
          label="作者名"
          value={config.author || ""}
          onChange={(v) => update("author", v)}
          placeholder="夏阳"
        />
        <Field
          label="GitHub 链接"
          value={config.github || ""}
          onChange={(v) => update("github", v)}
          placeholder="https://github.com/..."
        />
        <Field
          label="正文字号 (px)"
          value={config.fontSize || "17"}
          onChange={(v) => update("fontSize", v)}
          placeholder="17"
        />
      </div>

      {/* Password Change */}
      <PasswordSection />
    </motion.div>
  );
}

/* ─── Password Change Section ──────────────────────────── */

function PasswordSection() {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [pwMsg, setPwMsg] = useState<{ text: string; ok: boolean } | null>(null);
  const [pwLoading, setPwLoading] = useState(false);

  async function handleChangePassword(e: React.FormEvent) {
    e.preventDefault();
    setPwMsg(null);

    if (newPassword !== confirmPassword) {
      setPwMsg({ text: "两次输入的新密码不一致", ok: false });
      return;
    }

    setPwLoading(true);
    try {
      const res = await fetch("/api/auth/password", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ currentPassword, newPassword }),
      });
      const data = await res.json();
      if (res.ok) {
        setPwMsg({ text: "密码修改成功！", ok: true });
        setCurrentPassword("");
        setNewPassword("");
        setConfirmPassword("");
      } else {
        setPwMsg({ text: data.error || "修改失败", ok: false });
      }
    } catch {
      setPwMsg({ text: "网络错误，请重试", ok: false });
    }
    setPwLoading(false);
  }

  const inputClass =
    "w-full px-4 py-3 rounded-xl border border-border-medium bg-bg-primary/50 text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all duration-200";

  return (
    <div className="mt-8 glass rounded-2xl p-6 md:p-8">
      <h2 className="font-display text-lg font-semibold text-text-primary mb-1">
        修改密码
      </h2>
      <p className="text-sm text-text-tertiary mb-6">修改管理员登录密码</p>

      <form onSubmit={handleChangePassword} className="space-y-4 max-w-sm">
        <input
          type="password"
          value={currentPassword}
          onChange={(e) => setCurrentPassword(e.target.value)}
          placeholder="当前密码"
          className={inputClass}
          required
        />
        <input
          type="password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          placeholder="新密码（至少 6 位）"
          className={inputClass}
          required
          minLength={6}
        />
        <input
          type="password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          placeholder="确认新密码"
          className={inputClass}
          required
        />

        {pwMsg && (
          <motion.p
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className={`text-sm ${pwMsg.ok ? "text-green-600 dark:text-green-400" : "text-red-500"}`}
          >
            {pwMsg.text}
          </motion.p>
        )}

        <button
          type="submit"
          disabled={pwLoading}
          className="px-5 py-2.5 rounded-xl bg-accent text-white text-sm font-medium hover:bg-accent/90 active:scale-[0.98] transition-all duration-200 disabled:opacity-50"
        >
          {pwLoading ? "修改中..." : "修改密码"}
        </button>
      </form>
    </div>
  );
}

/* ─── Form Field ───────────────────────────────────────── */

function Field({
  label,
  value,
  onChange,
  placeholder,
  type = "text",
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  type?: string;
}) {
  const inputClass =
    "w-full px-4 py-3 rounded-xl border border-border-medium bg-bg-primary/50 text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all duration-200";

  return (
    <div>
      <label className="block text-sm font-medium text-text-secondary mb-2">
        {label}
      </label>
      {type === "textarea" ? (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          rows={3}
          className={`${inputClass} resize-y`}
        />
      ) : (
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className={inputClass}
        />
      )}
    </div>
  );
}
