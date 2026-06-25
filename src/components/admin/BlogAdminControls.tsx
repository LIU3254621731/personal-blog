"use client";

import { useEffect, useState } from "react";
import { InlineCreateButton, InlineEditButton, InlineDeleteButton, AdminActions } from "@/components/admin/InlineControls";

export function BlogAdminBar() {
  const [authed, setAuthed] = useState(false);
  useEffect(() => {
    fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {});
  }, []);
  if (!authed) return null;
  return <div className="mb-6"><InlineCreateButton href="/blog/new" label="新文章" /></div>;
}

export function BlogDetailAdminBar({ slug }: { slug: string }) {
  const [authed, setAuthed] = useState(false);
  useEffect(() => {
    fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {});
  }, []);
  if (!authed) return null;
  return <div className="mb-2"><InlineEditButton href={`/blog/edit/${slug}`} label="编辑此文章" /></div>;
}

export function BlogAdminActions({ postId, slug }: { postId: string; slug: string }) {
  const [authed, setAuthed] = useState(false);
  useEffect(() => {
    fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {});
  }, []);
  if (!authed) return null;

  async function handleDelete() {
    const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
    await fetch(`/api/posts/${postId}`, { method: "DELETE", headers: { "x-csrf-token": csrf } });
    window.location.reload();
  }

  return (
    <AdminActions>
      <InlineEditButton href={`/blog/edit/${slug}`} />
      <InlineDeleteButton onDelete={handleDelete} />
    </AdminActions>
  );
}
