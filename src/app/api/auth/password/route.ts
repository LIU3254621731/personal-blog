import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated, verifyCsrf, changePassword } from "@/lib/auth";
import { validateBody, changePasswordSchema } from "@/lib/validation";

export async function PUT(req: NextRequest) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }
  if (!(await verifyCsrf(req))) {
    return NextResponse.json({ error: "CSRF 校验失败" }, { status: 403 });
  }

  const parsed = await validateBody(req, changePasswordSchema);
  if (parsed instanceof NextResponse) return parsed;

  const result = await changePassword(parsed.currentPassword, parsed.newPassword);

  if (!result.success) {
    return NextResponse.json({ error: result.error }, { status: 400 });
  }

  return NextResponse.json({ success: true });
}
