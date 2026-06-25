import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated, verifyCsrf } from "@/lib/auth";
import { getSiteConfig, setSiteConfigs } from "@/lib/db";
import { validateBody, siteConfigSchema } from "@/lib/validation";

export async function GET() {
  return NextResponse.json(getSiteConfig());
}

export async function PUT(req: NextRequest) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }
  if (!(await verifyCsrf(req))) {
    return NextResponse.json({ error: "CSRF 校验失败" }, { status: 403 });
  }

  const parsed = await validateBody(req, siteConfigSchema);
  if (parsed instanceof NextResponse) return parsed;

  setSiteConfigs(parsed);
  return NextResponse.json({ success: true });
}
