"use client";

import { useState } from "react";
import { AdminLoginModal } from "@/components/admin/AdminLoginModal";

/**
 * A minimal admin entry point — a tiny dot in the footer.
 * Clicking it opens the admin login modal.
 */
export function AdminEntry() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="text-xs text-text-tertiary hover:text-accent transition-colors tracking-wider"
        title="管理后台"
      >
        ·
      </button>
      <AdminLoginModal open={open} onClose={() => setOpen(false)} />
    </>
  );
}
