"use client";

import { useEffect, useState } from "react";
import { InlineCreateButton } from "@/components/admin/InlineControls";

export function LearningAdminBar() {
  const [authed, setAuthed] = useState(false);
  useEffect(() => {
    fetch("/api/auth/check").then(r => r.json()).then(d => setAuthed(d.authenticated)).catch(() => {});
  }, []);
  if (!authed) return null;
  return <div className="mb-6"><InlineCreateButton href="/learning/new" label="新笔记" /></div>;
}
