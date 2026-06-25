"use client";

import { RoadmapAdminBar } from "@/components/admin/RoadmapAdminControls";
import { RoadmapActionsPortal } from "./RoadmapActionsPortal";

interface Goal {
  key: string;
}

export function RoadmapAdminControls({ goals }: { goals: Goal[] }) {
  return (
    <>
      <RoadmapAdminBar onAdd={() => { window.location.href = "/roadmap/new"; }} />
      <RoadmapActionsPortal goals={goals} />
    </>
  );
}
