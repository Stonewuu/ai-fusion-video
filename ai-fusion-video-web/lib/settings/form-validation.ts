import type { ApiConfigSaveReq } from "@/lib/api/ai-model";

export interface ConfigIssue { field: string; message: string }

export function httpAddressIssue(value: string, label: string): string | null {
  try {
    const url = new URL(value.trim());
    if ((url.protocol === "http:" || url.protocol === "https:") && url.hostname && !url.username && !url.password) return null;
  } catch { /* 返回明确的字段校验结果，不替换用户输入。 */ }
  return `${label}需填写完整的 http:// 或 https:// 地址，鉴权信息请放在专用字段。`;
}

export function apiConfigIssue(form: ApiConfigSaveReq, fields: { key: string; label: string; required?: boolean }[], noKey: boolean): ConfigIssue | null {
  if (!form.name.trim()) return { field: "api-config-name", message: "请填写配置名称。" };
  const values = form as unknown as Record<string, unknown>;
  if (noKey && form.platform === "openai_compatible" && !form.apiUrl?.trim())
    return { field: "api-field-apiUrl", message: "无需密钥的服务需填写具体的 API 地址。" };
  for (const field of fields) {
    if (field.required && !(field.key === "apiKey" && noKey && form.platform === "openai_compatible") && !String(values[field.key] ?? "").trim())
      return { field: `api-field-${field.key}`, message: `请填写${field.label}。` };
  }
  if (form.apiUrl?.trim() && form.platform !== "vertex_ai") {
    const message = httpAddressIssue(form.apiUrl, "API 地址");
    if (message) return { field: "api-field-apiUrl", message };
  }
  if (form.proxyType && form.proxyType !== "none") {
    if (!form.proxyHost?.trim()) return { field: "api-proxy-host", message: "启用代理后，请填写代理主机。" };
    if (!Number.isInteger(form.proxyPort) || !form.proxyPort || form.proxyPort < 1 || form.proxyPort > 65535)
      return { field: "api-proxy-port", message: "代理端口需为 1–65535 的整数。" };
    if (form.proxyPassword && !form.proxyUsername?.trim()) return { field: "api-proxy-user", message: "填写代理密码时也需要填写代理用户名。" };
  }
  return null;
}

export function skillFormIssue(form: { name: string; displayName: string; description: string; content: string }, originalName?: string): ConfigIssue | null {
  if (!form.displayName.trim()) return { field: "skill-display-name", message: "请填写 Skill 显示名称。" };
  if (Array.from(form.displayName.trim()).length > 64) return { field: "skill-display-name", message: "显示名称不能超过 64 个字符。" };
  const name = form.name.trim();
  const validName = originalName === name ? /^[a-z0-9][a-z0-9_-]{0,63}$/.test(name) : /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(name);
  if (!validName || name.length > 64)
    return { field: "skill-name", message: "调用名称请使用小写字母、数字和单个短横线，最长 64 位。" };
  if (!form.description.trim()) return { field: "skill-description", message: "请填写一句话描述，方便选择时识别用途。" };
  if (Array.from(form.description.trim()).length > 1024) return { field: "skill-description", message: "描述不能超过 1024 个字符。" };
  if (!form.content.trim()) return { field: "skill-content", message: "请填写 Skill 的工作步骤和要求。" };
  return null;
}

export function parseMcpLines(value: string): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [index, raw] of value.split("\n").entries()) {
    const line = raw.trim();
    if (!line) continue;
    const separator = line.indexOf(":");
    if (separator <= 0) throw new Error(`第 ${index + 1} 行格式错误，请使用「名称: 值」。`);
    const key = line.slice(0, separator).trim();
    if (!key || Object.hasOwn(result, key)) throw new Error(`第 ${index + 1} 行的名称为空或重复，请检查。`);
    Object.defineProperty(result, key, { value: line.slice(separator + 1).trim(), enumerable: true, writable: true, configurable: true });
  }
  return result;
}

export function mcpFormIssue(form: { name: string; url: string; headers: string; queryParams: string; timeoutSeconds: string; initializationTimeoutSeconds: string }): ConfigIssue | null {
  if (!form.name.trim()) return { field: "mcp-name", message: "请填写 MCP 名称。" };
  const addressError = httpAddressIssue(form.url, "服务地址");
  if (addressError) return { field: "mcp-url", message: addressError };
  for (const [field, value] of [["mcp-headers", form.headers], ["mcp-query", form.queryParams]]) {
    try { parseMcpLines(value); }
    catch (error) { return { field, message: `${field === "mcp-headers" ? "请求头" : "查询参数"}：${(error as Error).message}` }; }
  }
  for (const [field, value] of [["mcp-init-timeout", form.initializationTimeoutSeconds], ["mcp-timeout", form.timeoutSeconds]]) {
    const seconds = Number(value);
    if (!Number.isInteger(seconds) || seconds < 1 || seconds > 600)
      return { field, message: `${field === "mcp-init-timeout" ? "初始化" : "调用"}超时需为 1 到 600 秒之间的整数。` };
  }
  return null;
}
