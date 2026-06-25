import { NextRequest, NextResponse } from "next/server";
import { login } from "@/lib/auth";
import { validateBody, loginSchema } from "@/lib/validation";

export async function POST(req: NextRequest) {
  const parsed = await validateBody(req, loginSchema);
  if (parsed instanceof NextResponse) return parsed;

  const ok = await login(parsed.password);
  if (ok) return NextResponse.json({ success: true });
  return NextResponse.json({ error: "密码错误" }, { status: 401 });
}
