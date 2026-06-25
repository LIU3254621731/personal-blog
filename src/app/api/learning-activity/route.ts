import { NextResponse } from "next/server";
import { getLearningActivities } from "@/lib/db";

export async function GET() {
  const activities = getLearningActivities();
  return NextResponse.json({ activities });
}
