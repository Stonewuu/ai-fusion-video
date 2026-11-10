import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

const source = fs.readFileSync(new URL("../lib/settings/model-preset-filter.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText;
const loaded = { exports: {} };
vm.runInThisContext("(function(module,exports){" + compiled + "\n})")(loaded, loaded.exports);
const { filterModelPresets } = loaded.exports;
const presets = [
  { code: "claude", platform: "anthropic", modelType: 1 },
  { code: "gemini", platform: "gemini", modelType: 1, matchPlatforms: ["gemini", "vertex_ai", "vertex_ai_api_key"] },
  { code: "image", platform: "gemini", modelType: 2, matchPlatforms: ["gemini", "vertex_ai"] },
  { code: "flow", platform: "GoogleFlowReverseApi", modelType: 2, matchPlatforms: ["GoogleFlowReverseApi"] },
  { code: "sponsor", platform: "openai_compatible", modelType: 2, providerId: "infistar" },
];

test("Vertex 只显示对应类型的 Google 预设，不混入其他厂商或 Flow 别名", () => {
  assert.deepEqual(filterModelPresets(presets, 1, "vertex_ai").map(item => item.code), ["gemini"]);
  assert.deepEqual(filterModelPresets(presets, 2, "vertex_ai").map(item => item.code), ["image"]);
  assert.deepEqual(filterModelPresets(presets, 2, "GoogleFlowReverseApi").map(item => item.code), ["flow"]);
});

test("兼容渠道保留通用预设，赞助渠道只显示自身的专属生图预设", () => {
  assert.deepEqual(filterModelPresets(presets, 1, "openai_compatible").map(item => item.code), ["claude"]);
  assert.deepEqual(filterModelPresets(presets, 2, "openai_compatible", { id: "infistar", quickSetup: true }).map(item => item.code), ["sponsor"]);
});
