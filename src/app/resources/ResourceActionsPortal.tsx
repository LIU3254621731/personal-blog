"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { ResourceAdminActions } from "@/components/admin/ResourceAdminControls";

interface Resource {
  key: string;
}

export function ResourceActionsPortal({ resources }: { resources: Resource[] }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <>
      {resources.map((r) => {
        const el = document.getElementById(`resource-actions-${r.key}`);
        if (!el) return null;
        return createPortal(
          <ResourceAdminActions
            key={r.key}
            resourceKey={r.key.replace("resource_", "")}
            onDelete={async () => {
              const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
              await fetch(`/api/site-config`, {
                method: "DELETE",
                headers: { "x-csrf-token": csrf, "content-type": "application/json" },
                body: JSON.stringify({ key: r.key }),
              });
              window.location.reload();
            }}
          />,
          el,
        );
      })}
    </>
  );
}
