"use client";

import { useEffect, useState } from "react";

export default function SettingsPage() {
  const [config, setConfig] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    fetch("/api/site-config").then(r => r.json()).then(setConfig);
  }, []);

  async function handleSave() {
    await fetch("/api/site-config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  function update(key: string, value: string) {
    setConfig(prev => ({ ...prev, [key]: value }));
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0 }}>站点设置</h1>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {saved && <span style={{ fontSize: 13, color: "#16a34a" }}>已保存</span>}
          <button onClick={handleSave}
            style={{ padding: "10px 24px", borderRadius: 10, background: "#1c1c1e", color: "#fff", border: "none", fontSize: 14, fontWeight: 600, cursor: "pointer" }}>
            保存设置
          </button>
        </div>
      </div>
      <div style={{ background: "#fff", borderRadius: 14, padding: 28, border: "1px solid #e8e6e1" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <Field label="站点标题" value={config.title || ""} onChange={v => update("title", v)} placeholder="Liu Wenlin" />
          <Field label="副标题" value={config.subtitle || ""} onChange={v => update("subtitle", v)} placeholder="思考与创造" />
          <Field label="站点描述" value={config.description || ""} onChange={v => update("description", v)} placeholder="记录关于代码、AI 和探索的个人空间。" type="textarea" />
          <Field label="作者名" value={config.author || ""} onChange={v => update("author", v)} placeholder="Liu Wenlin" />
          <Field label="GitHub 链接" value={config.github || ""} onChange={v => update("github", v)} placeholder="https://github.com/..." />
          <Field label="正文字号 (px)" value={config.fontSize || "17"} onChange={v => update("fontSize", v)} placeholder="17" />
        </div>
      </div>
    </div>
  );
}

function Field({ label, value, onChange, placeholder, type = "text" }: {
  label: string; value: string; onChange: (v: string) => void; placeholder: string; type?: string;
}) {
  return (
    <div>
      <label style={{ display: "block", fontSize: 13, fontWeight: 600, color: "#555", marginBottom: 6 }}>{label}</label>
      {type === "textarea" ? (
        <textarea value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} rows={3}
          style={{ width: "100%", padding: "10px 14px", borderRadius: 10, border: "1px solid #e0ded9", fontSize: 14, outline: "none", boxSizing: "border-box", resize: "vertical", fontFamily: "inherit" }} />
      ) : (
        <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
          style={{ width: "100%", padding: "10px 14px", borderRadius: 10, border: "1px solid #e0ded9", fontSize: 14, outline: "none", boxSizing: "border-box" }} />
      )}
    </div>
  );
}
