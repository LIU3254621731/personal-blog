"use client";

import { ResourceAdminBar } from "@/components/admin/ResourceAdminControls";
import { ResourceActionsPortal } from "./ResourceActionsPortal";

interface Resource {
  key: string;
}

export function ResourceAdminControls({ resources }: { resources: Resource[] }) {
  return (
    <>
      <ResourceAdminBar />
      <ResourceActionsPortal resources={resources} />
    </>
  );
}
