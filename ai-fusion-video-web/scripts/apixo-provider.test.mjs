import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import ts from "typescript";

function load(relative, modules = {}) {
  const source = readFileSync(new URL(relative, import.meta.url), "utf8");
  const compiled = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText;
  const loaded = { exports: {} };
  vm.runInThisContext(`(function(module,exports,require){${compiled}\n})`)(loaded, loaded.exports, (name) => {
    if (!(name in modules)) {
      throw new Error(`未配置测试模块：${name}`);
    }
    return modules[name];
  });
  return loaded.exports;
}

const { orderProviderPresets } = load("../app/(dashboard)/settings/ai-models/_components/provider-preset-order.ts");
const { imageOutputCapabilities, resolutionChoices } = load("../lib/image-output-sizes.ts");
const { videoInputState, videoRatioOptions } = load("../lib/video-model-rules.ts");
const { resolveGenerationCapabilities } = load("../lib/generation-capabilities.ts", {
  "@/lib/video-model-rules": load("../lib/video-model-rules.ts"),
  "@/lib/image-output-sizes": load("../lib/image-output-sizes.ts"),
});

test("APIXO 排在已有赞助商之后，普通提供商仍位于赞助商之后", () => {
  const input = [
    { id: "apixo", sponsored: true }, { id: "openai", sponsored: false },
    { id: "erchun", sponsored: true }, { id: "infistar", sponsored: true },
    { id: "metaso", sponsored: true },
  ];
  assert.deepEqual(orderProviderPresets(input).map((provider) => provider.id),
    ["infistar", "metaso", "erchun", "apixo", "openai"]);
  assert.equal(input[0].id, "apixo");
});

test("Sora 未开放分辨率参数时，生成面板不补出 720p 和 1080p", () => {
  const capabilities = resolveGenerationCapabilities({
    modelType: 3,
    generationConfig: { supportedResolutions: [], supportedAspectRatios: ["16:9", "9:16"] },
  }, []);
  assert.deepEqual(capabilities.supportedResolutions, []);
});

test("自动画面比例保留 APIXO 的分辨率档位，并保持像素尺寸模型的自动选项", () => {
  const capabilities = imageOutputCapabilities({
    supportedAspectRatios: ["auto", "1:1"],
    supportedResolutions: ["1k", "1.5k", "2k"],
    supportAutoSize: true,
  });
  assert.deepEqual(capabilities.supportedAspectRatios, ["auto", "1:1"]);
  assert.deepEqual(resolutionChoices(capabilities, "auto"), ["1k", "1.5k", "2k"]);
  assert.deepEqual(resolutionChoices({
    supportedResolutions: ["auto", "1k"],
    imageSizes: { "1k": { "1:1": "1024x1024" } },
    supportAutoSize: true,
  }, "auto"), ["auto"]);
});

test("Seedance 2.5 添加首帧或首尾帧后只显示自适应比例，文生视频保留固定比例", () => {
  const presets = JSON.parse(readFileSync(new URL(
    "../../ai-fusion-video/src/main/resources/model-presets/apixo/video.json", import.meta.url), "utf8"));
  for (const code of ["seedance-2-5", "seedance-2-5-flash"]) {
    const preset = presets.find((item) => item.modelCode === code);
    const capabilities = resolveGenerationCapabilities({ modelType: 3, generationConfig: preset.config }, []);
    assert.deepEqual(videoRatioOptions(capabilities.supportedAspectRatios, capabilities.videoRules,
      videoInputState({ firstFrame: ["https://example.com/first.png"] })), ["adaptive"]);
    assert.deepEqual(videoRatioOptions(capabilities.supportedAspectRatios, capabilities.videoRules,
      videoInputState({ firstFrame: ["https://example.com/first.png"], lastFrame: ["https://example.com/last.png"] })), ["adaptive"]);
    assert.ok(videoRatioOptions(capabilities.supportedAspectRatios, capabilities.videoRules,
      videoInputState({})).includes("16:9"));
  }
});
