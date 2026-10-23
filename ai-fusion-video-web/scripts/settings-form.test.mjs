import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

const file = new URL("../lib/settings/form-validation.ts", import.meta.url);
const compiled = ts.transpileModule(fs.readFileSync(file, "utf8"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText;
const loaded = { exports: {} };
vm.runInThisContext(`(function(module,exports){${compiled}\n})`)(loaded, loaded.exports);
const { apiConfigIssue, skillFormIssue, mcpFormIssue, parseMcpLines, httpAddressIssue } = loaded.exports;
const cloudFields = [{ key: "apiKey", label: "API 密钥", required: true }];

test("API config requires a key unless an explicit keyless local endpoint is supplied", () => {
  const config = { name: "Local", platform: "openai_compatible" };
  assert.equal(apiConfigIssue(config, cloudFields, false)?.field, "api-field-apiKey");
  assert.equal(apiConfigIssue(config, cloudFields, true)?.field, "api-field-apiUrl");
  assert.equal(apiConfigIssue({ ...config, apiUrl: "http://localhost:18096" }, cloudFields, true), null);
  assert.equal(apiConfigIssue({ ...config, platform: "gemini" }, cloudFields, true)?.field, "api-field-apiKey");
});

test("API addresses reject malformed, non HTTP and embedded credentials while Vertex regions remain valid", () => {
  for (const value of ["example.com", "file:///tmp/model", "https://user:secret@example.com"]) {
    assert.ok(httpAddressIssue(value, "服务地址"));
  }
  assert.equal(httpAddressIssue("https://example.com/v1", "服务地址"), null);
  assert.equal(apiConfigIssue({ name: "Vertex", platform: "vertex_ai", appId: "project", apiUrl: "us-central1" }, [{ key: "appId", label: "项目 ID", required: true }], false), null);
});

test("API proxy validates port and credentials only when enabled", () => {
  const config = { name: "Proxy", platform: "ollama", proxyType: "http", proxyHost: "localhost" };
  for (const proxyPort of [undefined, 0, 1.5, 65536]) {
    assert.equal(apiConfigIssue({ ...config, proxyPort }, [], false)?.field, "api-proxy-port");
  }
  assert.equal(apiConfigIssue({ ...config, proxyPort: 7890, proxyPassword: "secret" }, [], false)?.field, "api-proxy-user");
  assert.equal(apiConfigIssue({ ...config, proxyPort: 7890 }, [], false), null);
  assert.equal(apiConfigIssue({ ...config, proxyType: "none" }, [], false), null);
});

test("Skill names and code point limits agree with the server including existing legacy names", () => {
  const skill = { name: "scene-review", displayName: "连续性检查", description: "检查镜头", content: "# 工作步骤" };
  assert.equal(skillFormIssue(skill), null);
  for (const name of ["Scene-Review", "scene--review", "scene_review", "-review", "review-"]) {
    assert.equal(skillFormIssue({ ...skill, name })?.field, "skill-name");
  }
  assert.equal(skillFormIssue({ ...skill, name: "old_skill" }, "old_skill"), null);
  assert.equal(skillFormIssue({ ...skill, displayName: "😀".repeat(64) }), null);
  assert.equal(skillFormIssue({ ...skill, displayName: "😀".repeat(65) })?.field, "skill-display-name");
  assert.equal(skillFormIssue({ ...skill, description: "文".repeat(1025) })?.field, "skill-description");
});

test("MCP line parsing preserves colons, rejects duplicates and never echoes secret values in errors", () => {
  assert.deepEqual(parseMcpLines("Authorization: Bearer test\nurl: https://example.com\n\n"), { Authorization: "Bearer test", url: "https://example.com" });
  assert.throws(() => parseMcpLines("invalid-secret-value"), error => !error.message.includes("invalid-secret-value") && error.message.includes("第 1 行"));
  assert.throws(() => parseMcpLines("tenant: one\ntenant: two"), /第 2 行/);
  assert.equal(Object.getPrototypeOf(parseMcpLines("__proto__: value")), Object.prototype);
});

test("MCP reports the first invalid field without submitting remote configuration", () => {
  const form = { name: "test", url: "https://example.com/mcp", headers: "", queryParams: "", timeoutSeconds: "120", initializationTimeoutSeconds: "30" };
  assert.equal(mcpFormIssue(form), null);
  assert.equal(mcpFormIssue({ ...form, name: "" })?.field, "mcp-name");
  assert.equal(mcpFormIssue({ ...form, url: "mcp.example.com" })?.field, "mcp-url");
  assert.equal(mcpFormIssue({ ...form, headers: "secret" })?.field, "mcp-headers");
  assert.equal(mcpFormIssue({ ...form, timeoutSeconds: "601" })?.field, "mcp-timeout");
  assert.equal(mcpFormIssue({ ...form, initializationTimeoutSeconds: "0" })?.field, "mcp-init-timeout");
});

test("Renaming a storage config does not submit unchanged empty location or credential fields", () => {
  const source = fs.readFileSync(new URL("../lib/settings/storage-form.ts", import.meta.url), "utf8");
  const code = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText;
  const storage = { exports: {} };
  vm.runInThisContext(`(function(module,exports){${code}\n})`)(storage, storage.exports);
  const original = { id: 1, name: "旧名称", type: "local", basePath: "/data", endpoint: "", bucketName: "", region: "", secretKey: "" };
  assert.deepEqual(storage.exports.storageUpdatePayload({ ...original, name: "新名称" }, original), { id: 1, name: "新名称" });
  assert.deepEqual(storage.exports.storageUpdatePayload({ ...original, basePath: "/new-data" }, original), { id: 1, name: "旧名称", basePath: "/new-data" });
  const s3 = { ...original, type: "s3", customDomain: "https://cdn.example.com" };
  assert.deepEqual(storage.exports.storageUpdatePayload({ ...s3, customDomain: "" }, s3), { id: 1, name: "旧名称", customDomain: "" });
});
