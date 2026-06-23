import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getSiteConfig, setSiteConfigs } from "@/lib/db";

export async function GET() {
  const config = getSiteConfig();
  return NextResponse.json(config);
}

export async function PUT(req: NextRequest) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }
  const data = await req.json();
  setSiteConfigs(data);
  return NextResponse.json({ success: true });
}
