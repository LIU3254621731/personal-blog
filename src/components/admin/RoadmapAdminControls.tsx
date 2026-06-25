"use client";
import { useEffect, useState } from "react";
import { InlineCreateButton, InlineEditButton, InlineDeleteButton, AdminActions } from "./InlineControls";

export function RoadmapAdminBar({ onAdd }: { onAdd: () => void }) {
  const [authed, setAuthed] = useState(false);
  useEffect(() => { fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {}); }, []);
  if (!authed) return null;
  return <div className="mb-6"><button onClick={onAdd} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent text-white text-xs font-medium">+ 新目标</button></div>;
}

export function RoadmapAdminActions({ resourceKey, onDelete }: { resourceKey: string; onDelete: () => void }) {
  const [authed, setAuthed] = useState(false);
  useEffect(() => { fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {}); }, []);
  if (!authed) return null;
  return <AdminActions><InlineEditButton href={`/roadmap/edit/${resourceKey}`} /><InlineDeleteButton onDelete={onDelete} /></AdminActions>;
}
