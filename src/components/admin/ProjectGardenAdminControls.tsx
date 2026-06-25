"use client";

import { useEffect, useState } from "react";
import { InlineCreateButton, InlineEditButton, InlineDeleteButton, AdminActions } from "@/components/admin/InlineControls";

export function ProjectAdminBar() {
  const [authed, setAuthed] = useState(false);
  useEffect(() => {
    fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {});
  }, []);
  if (!authed) return null;
  return <div className="mb-6"><InlineCreateButton href="/projects/new" label="新项目" /></div>;
}

export function ProjectAdminActions({ projectId }: { projectId: string }) {
  const [authed, setAuthed] = useState(false);
  useEffect(() => {
    fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {});
  }, []);
  if (!authed) return null;

  async function handleDelete() {
    const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
    await fetch(`/api/projects/${projectId}`, { method: "DELETE", headers: { "x-csrf-token": csrf } });
    window.location.reload();
  }

  return (
    <AdminActions>
      <InlineEditButton href={`/projects/edit/${projectId}`} />
      <InlineDeleteButton onDelete={handleDelete} />
    </AdminActions>
  );
}

export function GardenAdminBar() {
  const [authed, setAuthed] = useState(false);
  useEffect(() => {
    fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {});
  }, []);
  if (!authed) return null;
  return <div className="mb-6"><InlineCreateButton href="/garden/new" label="新种子" /></div>;
}

export function GardenAdminActions({ entryId }: { entryId: string }) {
  const [authed, setAuthed] = useState(false);
  useEffect(() => {
    fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {});
  }, []);
  if (!authed) return null;

  async function handleDelete() {
    const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
    await fetch(`/api/garden/${entryId}`, { method: "DELETE", headers: { "x-csrf-token": csrf } });
    window.location.reload();
  }

  return (
    <AdminActions>
      <InlineEditButton href={`/garden/edit/${entryId}`} />
      <InlineDeleteButton onDelete={handleDelete} />
    </AdminActions>
  );
}
