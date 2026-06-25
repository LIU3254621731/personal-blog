"use client";

import { useState, useEffect } from "react";
import { InlineCreateButton, InlineEditButton, InlineDeleteButton, AdminActions } from "./InlineControls";
import { AdminFormModal } from "./AdminFormModal";

export function SiteConfigAdminBar({
  prefix,
  label,
  fields,
  onSave,
}: {
  prefix: string;
  label: string;
  fields: { key: string; label: string; type?: "text" | "textarea" | "number" }[];
  onSave: () => void;
}) {
  const [authed, setAuthed] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {});
  }, []);

  if (!authed) return null;

  return (
    <div className="mb-6">
      <button
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent text-white text-xs font-medium hover:bg-accent/90 active:scale-[0.98] transition-all"
      >
        +{label}
      </button>
      <AdminFormModal
        open={open}
        onClose={() => setOpen(false)}
        title={`新建${label}`}
        fields={fields}
        onSave={async (data) => {
          const key = `${prefix}_${Date.now().toString(36)}`;
          const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
          const url = `/api/${prefix}`;
          await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json", "x-csrf-token": csrf },
            body: JSON.stringify({ key, ...data }),
          });
          setOpen(false);
          onSave();
        }}
      />
    </div>
  );
}

export function SiteConfigAdminActions({
  resourceKey,
  prefix,
  fields,
  currentData,
  onDelete,
  onSave,
}: {
  resourceKey: string;
  prefix: string;
  fields: { key: string; label: string; type?: "text" | "textarea" | "number" }[];
  currentData: Record<string, any>;
  onDelete: () => void;
  onSave: () => void;
}) {
  const [authed, setAuthed] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {});
  }, []);

  if (!authed) return null;

  async function handleDelete() {
    const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
    await fetch(`/api/${prefix}/${encodeURIComponent(resourceKey)}`, {
      method: "DELETE",
      headers: { "x-csrf-token": csrf },
    });
    onDelete();
  }

  return (
    <>
      <AdminActions>
        <button onClick={() => setOpen(true)} className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs text-text-tertiary hover:text-accent hover:bg-accent-light/30 transition-all">
          编辑
        </button>
        <InlineDeleteButton onDelete={handleDelete} />
      </AdminActions>
      <AdminFormModal
        open={open}
        onClose={() => setOpen(false)}
        title="编辑"
        fields={fields}
        initialData={currentData}
        onSave={async (data) => {
          const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
          await fetch(`/api/${prefix}/${encodeURIComponent(resourceKey)}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json", "x-csrf-token": csrf },
            body: JSON.stringify(data),
          });
          setOpen(false);
          onSave();
        }}
      />
    </>
  );
}
