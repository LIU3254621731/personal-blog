import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated, changePassword } from "@/lib/auth";
import { validateBody, changePasswordSchema } from "@/lib/validation";

export async function PUT(req: NextRequest) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }

  const parsed = await validateBody(req, changePasswordSchema);
  if (parsed instanceof NextResponse) return parsed;

  const result = await changePassword(parsed.currentPassword, parsed.newPassword);

  if (!result.success) {
    return NextResponse.json({ error: result.error }, { status: 400 });
  }

  return NextResponse.json({ success: true });
}
