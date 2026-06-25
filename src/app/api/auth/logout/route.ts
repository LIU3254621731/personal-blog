import { NextResponse } from "next/server";
import { isAuthenticated, logout } from "@/lib/auth";

export async function POST() {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }
  await logout();
  return NextResponse.json({ success: true });
}
