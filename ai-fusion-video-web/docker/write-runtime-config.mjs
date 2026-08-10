import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const rawApiBaseUrl = (
  process.env.PUBLIC_API_URL ?? ""
).trim();

let apiBaseUrl = rawApiBaseUrl.replace(/\/+$/, "");

if (apiBaseUrl) {
  let parsed;
  try {
    parsed = new URL(apiBaseUrl);
  } catch {
    throw new Error(`PUBLIC_API_URL 不是合法 URL: ${apiBaseUrl}`);
  }

  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error("PUBLIC_API_URL 仅支持 http:// 或 https:// 地址");
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error("PUBLIC_API_URL 不能包含认证信息、查询参数或锚点");
  }
  if (parsed.pathname.replace(/\/+$/, "").endsWith("/api")) {
    throw new Error("PUBLIC_API_URL 应填写后端根地址，不能包含末尾 /api");
  }
}

const config = Object.freeze({ apiBaseUrl });
const source = `window.__FUSION_RUNTIME_CONFIG__ = Object.freeze(${JSON.stringify(config)});\n`;
const outputPath = fileURLToPath(
  new URL("../public/runtime-config.js", import.meta.url),
);

writeFileSync(outputPath, source, "utf8");
console.log(
  `[runtime-config] API 地址: ${apiBaseUrl || "同源 /api（推荐）"}`,
);
