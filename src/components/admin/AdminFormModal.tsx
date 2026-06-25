"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Save } from "lucide-react";

interface Field {
  key: string;
  label: string;
  type?: "text" | "textarea" | "number";
}

function toStr(v: any): string {
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.join(", ");
  if (v != null) return String(v);
  return "";
}

export function AdminFormModal({
  open, onClose, title, fields, initialData, onSave,
}: {
  open: boolean; onClose: () => void; title: string;
  fields: Field[]; initialData?: Record<string, any>;
  onSave: (data: Record<string, any>) => Promise<void>;
}) {
  const [data, setData] = useState<Record<string, any>>({});
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (open) setData(initialData ? { ...initialData } : {});
  }, [open]);

  const ic = "w-full px-4 py-2.5 rounded-xl border border-border-medium bg-bg-primary/50 text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all duration-200";

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-[60] bg-black/20 backdrop-blur-sm" onClick={onClose} />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 8 }} animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 8 }} transition={{ duration: 0.2 }}
            className="fixed inset-0 z-[61] flex items-center justify-center p-4">
            <div className="glass-strong rounded-2xl p-6 w-full max-w-md relative" onClick={e => e.stopPropagation()}>
              <button onClick={onClose} className="absolute top-4 right-4 p-1.5 rounded-lg text-text-tertiary hover:text-text-primary transition-all">
                <X size={18} />
              </button>
              <h2 className="font-display text-lg font-semibold text-text-primary mb-5">{title}</h2>
              <div className="space-y-4">
                {fields.map(f => {
                  const val = toStr(data[f.key]);
                  return (
                    <div key={f.key}>
                      <label className="block text-xs font-medium text-text-secondary mb-1.5">{f.label}</label>
                      {f.type === "textarea" ? (
                        <textarea value={val} onChange={e => setData(d => ({ ...d, [f.key]: e.target.value }))}
                          rows={3} className={ic} placeholder={f.label} />
                      ) : f.type === "number" ? (
                        <input type="number" value={val} onChange={e => setData(d => ({ ...d, [f.key]: parseInt(e.target.value) || 0 }))}
                          className={ic} placeholder={f.label} min={0} max={100} />
                      ) : (
                        <input value={val} onChange={e => setData(d => ({ ...d, [f.key]: e.target.value }))}
                          className={ic} placeholder={f.label} />
                      )}
                    </div>
                  );
                })}
              </div>
              <button
                onClick={async () => { setSaving(true); await onSave(data); setSaving(false); }}
                disabled={saving}
                className="mt-5 w-full py-2.5 rounded-xl bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-all disabled:opacity-50 flex items-center justify-center gap-2">
                <Save size={15} />{saving ? "保存中..." : "保存"}
              </button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
