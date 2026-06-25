"use client";
import { useEffect, useState } from "react";
import { InlineCreateButton, InlineEditButton, InlineDeleteButton, AdminActions } from "./InlineControls";

export function ResourceAdminBar() {
  const [authed, setAuthed] = useState(false);
  useEffect(() => { fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {}); }, []);
  if (!authed) return null;
  return <div className="mb-6"><InlineCreateButton href="/resources/new" label="新资源" /></div>;
}

export function ResourceAdminActions({ resourceKey, onDelete }: { resourceKey: string; onDelete: () => void }) {
  const [authed, setAuthed] = useState(false);
  useEffect(() => { fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {}); }, []);
  if (!authed) return null;
  return (
    <AdminActions>
      <InlineEditButton href={`/resources/edit/${resourceKey}`} />
      <InlineDeleteButton onDelete={onDelete} />
    </AdminActions>
  );
}
