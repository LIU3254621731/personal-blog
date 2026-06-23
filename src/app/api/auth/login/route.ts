import { NextRequest, NextResponse } from "next/server";
import { login } from "@/lib/auth";

export async function POST(req: NextRequest) {
  const { password } = await req.json();
  const ok = await login(password);
  if (ok) return NextResponse.json({ success: true });
  return NextResponse.json({ error: "密码错误" }, { status: 401 });
}
