import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { writeFile, mkdir } from "fs/promises";
import path from "path";
import { ALLOWED_IMAGE_TYPES, MAX_UPLOAD_SIZE } from "@/lib/validation";

export async function POST(req: NextRequest) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }

  let formData: FormData;
  try {
    formData = await req.formData();
  } catch {
    return NextResponse.json({ error: "请求格式错误" }, { status: 400 });
  }

  const file = formData.get("file") as File | null;
  if (!file || typeof file === "string") {
    return NextResponse.json({ error: "未上传文件" }, { status: 400 });
  }

  // Validate file type
  if (!ALLOWED_IMAGE_TYPES.includes(file.type as (typeof ALLOWED_IMAGE_TYPES)[number])) {
    return NextResponse.json(
      { error: `不支持的文件类型: ${file.type}` },
      { status: 400 },
    );
  }

  // Validate file size
  if (file.size > MAX_UPLOAD_SIZE) {
    return NextResponse.json(
      { error: `文件大小不能超过 ${MAX_UPLOAD_SIZE / 1024 / 1024} MB` },
      { status: 400 },
    );
  }

  const bytes = await file.arrayBuffer();
  const buffer = Buffer.from(bytes);
  const safeExt = path.extname(file.name).replace(/[^a-zA-Z0-9.]/g, "") || ".png";
  const filename = `${Date.now()}-${Math.random().toString(36).substring(2, 8)}${safeExt}`;
  const uploadDir = path.join(process.cwd(), "public", "uploads");
  await mkdir(uploadDir, { recursive: true });
  await writeFile(path.join(uploadDir, filename), buffer);

  const url = `/uploads/${filename}`;
  return NextResponse.json({ url, filename });
}
