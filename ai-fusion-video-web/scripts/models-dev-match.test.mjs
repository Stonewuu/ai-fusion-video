import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

const source = fs.readFileSync(new URL("../lib/settings/models-dev-match.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText;
const loaded = { exports: {} };
vm.runInThisContext(`(function(module,exports){${compiled}\n})`)(loaded, loaded.exports);
const { applyModelsDevMatch, findOfficialGooglePreset } = loaded.exports;
const form = { name: "渠道模型", code: "my-model-id", modelProtocol: "google_flow", modelType: 2, apiConfigId: 8,
  capabilityPresetCode: "flow-preset", config: '{"customSize":true}', contextWindow: 100,
  multimodalInputTypes: [], multimodalInputTransports: {}, supportReasoning: true, reasoningEffortLevels: ["high"] };
const candidate = { providerId: "google", providerName: "Google", modelId: "gemini-3.1-flash-image", modelType: 2,
  canonicalModelId: "google/gemini-3.1-flash-image", contextWindow: 131072, outputTokenLimit: 32768,
  supportReasoning: null, reasoningEffortLevels: null, inputModalities: null, updatedAt: "2026-05-28" };

test("应用联网能力时保留调用标识、渠道预设与自定义参数", () => {
  const next = applyModelsDevMatch(form, candidate, "2026-11-06T00:00:00Z");
  assert.equal(next.code, "my-model-id");
  assert.equal(next.apiConfigId, 8);
  assert.equal(next.modelProtocol, "google_flow");
  assert.equal(next.capabilityPresetCode, "flow-preset");
  assert.equal(next.contextWindow, 131072);
  assert.equal(next.supportReasoning, true);
  assert.equal(JSON.parse(next.config).customSize, true);
  assert.equal(JSON.parse(next.config).catalogMatch.providerId, "google");
  assert.equal(form.contextWindow, 100);
});

test("官方 Google 预设只在对应渠道匹配，Flow 保持自己的配置", () => {
  const preset = { code: "google:gemini-3.1-flash-image", modelCode: candidate.modelId, modelType: 2,
    official: true, matchPlatforms: ["gemini", "vertex_ai", "vertex_ai_api_key"], contextWindow: 131072 };
  assert.equal(findOfficialGooglePreset(candidate, [preset], "GoogleFlowReverseApi"), undefined);
  assert.equal(findOfficialGooglePreset(candidate, [preset], "vertex_ai"), preset);
  assert.equal(findOfficialGooglePreset({ ...candidate, canonicalModelId: "google/gemini-3.1-flash-image-preview" }, [preset], "vertex_ai"), undefined);
});

test("明确的不支持覆盖旧值，未提供的能力保留原值，输入类型去重", () => {
  const text = { ...form, modelType: 1, multimodalInputTypes: ["image"], multimodalInputTransports: { image: ["url"] } };
  const next = applyModelsDevMatch(text, { ...candidate, modelType: 1, supportReasoning: false,
    inputModalities: ["text", "image", "pdf", "file"] }, "now");
  assert.equal(next.supportReasoning, false);
  assert.deepEqual(next.reasoningEffortLevels, []);
  assert.deepEqual(next.multimodalInputTypes, ["image", "file"]);
  assert.deepEqual(next.multimodalInputTransports.image, ["url"]);
  assert.deepEqual(next.multimodalInputTransports.file, ["base64"]);
  assert.throws(() => applyModelsDevMatch({ ...form, config: "invalid" }, candidate, "now"), /JSON 格式/);
});
