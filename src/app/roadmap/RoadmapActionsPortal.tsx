"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { RoadmapAdminActions } from "@/components/admin/RoadmapAdminControls";

interface Goal {
  key: string;
}

export function RoadmapActionsPortal({ goals }: { goals: Goal[] }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <>
      {goals.map((goal) => {
        const el = document.getElementById(`roadmap-actions-${goal.key}`);
        if (!el) return null;
        return createPortal(
          <RoadmapAdminActions
            key={goal.key}
            resourceKey={goal.key}
            onDelete={async () => {
              const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
              const rawKey = goal.key.replace("roadmap_", "");
              await fetch(`/api/site-config`, {
                method: "DELETE",
                headers: { "x-csrf-token": csrf, "content-type": "application/json" },
                body: JSON.stringify({ key: goal.key }),
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
