import { readFile } from "node:fs/promises";
import { join } from "node:path";

export const runtime = "nodejs";
// Next standalone 在项目根目录运行，依赖静态资源由 outputFileTracingIncludes 明确打包。
const root = join(process.cwd(), "node_modules", "pdfjs-dist");
const families: Record<string, { extensions: string[]; mime: string }> = {
  build: { extensions: ["mjs"], mime: "text/javascript" },
  cmaps: { extensions: ["bcmap"], mime: "application/octet-stream" },
  standard_fonts: {
    extensions: ["pfb", "ttf"],
    mime: "application/octet-stream",
  },
  wasm: { extensions: ["wasm"], mime: "application/wasm" },
};

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ family: string; name: string }> },
) {
  const { family, name } = await params;
  const spec = Object.hasOwn(families, family) ? families[family] : null;
  const match = /^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\.([a-z]+)$/.exec(name);
  if (family === "build" && name !== "pdf.worker.min.mjs")
    return new Response("Not found", { status: 404 });
  if (!spec || !match || !spec.extensions.includes(match[1]))
    return new Response("Not found", { status: 404 });
  try {
    const bytes = await readFile(join(root, family, name));
    return new Response(new Uint8Array(bytes), {
      headers: {
        "Content-Type": spec.mime,
        "Cache-Control": "public, max-age=86400",
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT")
      return new Response("Not found", { status: 404 });
    throw error;
  }
}
