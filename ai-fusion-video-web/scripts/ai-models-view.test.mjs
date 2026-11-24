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

const aiModelModuleStub = {
  MODEL_TYPE_OPTIONS: [
    { value: 1, label: "文字生成" },
    { value: 2, label: "图像生成" },
    { value: 3, label: "视频生成" },
    { value: 4, label: "音频生成" },
  ],
  PLATFORM_LABELS: {
    openai_compatible: "OpenAI / 兼容接入",
    dashscope: "DashScope",
  },
};

const {
  filterApiConfigsAndModels,
  resolveDefaultModelsByType,
  filterRemoteModels,
} = load("../app/(dashboard)/settings/ai-models/_components/ai-models-filter.ts", {
  "@/lib/api/ai-model": aiModelModuleStub,
});

const configs = [
  { id: 1, name: "主中转站", platform: "openai_compatible", status: 1 },
  { id: 2, name: "阿里百炼", platform: "dashscope", status: 0 },
];

const models = [
  { id: 11, apiConfigId: 1, name: "Claude 4", code: "claude-4", modelType: 1, status: 1, defaultModel: true },
  { id: 12, apiConfigId: 1, name: "GPT Image", code: "gpt-image-1", modelType: 2, status: 1, defaultModel: false },
  { id: 13, apiConfigId: 1, name: "旧版对话", code: "legacy-chat", modelType: 1, status: 0, defaultModel: false },
  { id: 21, apiConfigId: 2, name: "万相视频", code: "wan-2.6", modelType: 3, status: 1, defaultModel: true },
];

test("搜索模型关键词时仅展示命中的模型并计算隐藏数，搜服务名时保留该服务全部模型", () => {
  const byModel = filterApiConfigsAndModels(configs, models, {
    query: "claude",
    modelType: 0,
    status: "all",
  });
  assert.equal(byModel.length, 1);
  assert.deepEqual(byModel[0].visibleModels.map((item) => item.id), [11]);
  assert.equal(byModel[0].hiddenModelCount, 2);

  const byConfig = filterApiConfigsAndModels(configs, models, {
    query: "主中转站",
    modelType: 0,
    status: "all",
  });
  assert.equal(byConfig.length, 1);
  assert.equal(byConfig[0].matchedByConfig, true);
  assert.equal(byConfig[0].visibleModels.length, 3);
  assert.equal(byConfig[0].hiddenModelCount, 0);
});

test("按模型类型与启停状态筛选时正确识别服务停用对模型可用性的影响", () => {
  const enabledImages = filterApiConfigsAndModels(configs, models, {
    query: "",
    modelType: 2,
    status: "enabled",
  });
  assert.equal(enabledImages.length, 1);
  assert.deepEqual(enabledImages[0].visibleModels.map((item) => item.id), [12]);

  const disabledEntries = filterApiConfigsAndModels(configs, models, {
    query: "",
    modelType: 0,
    status: "disabled",
  });
  assert.deepEqual(disabledEntries.map((entry) => entry.config.id), [1, 2]);
  assert.deepEqual(disabledEntries[0].visibleModels.map((item) => item.id), [13]);
  assert.deepEqual(disabledEntries[1].visibleModels.map((item) => item.id), [21]);
});

test("全局默认模型槽位识别停用状态并仅提供可用候选模型", () => {
  const slots = resolveDefaultModelsByType(models, configs);
  const textSlot = slots.find((slot) => slot.modelType === 1);
  const videoSlot = slots.find((slot) => slot.modelType === 3);

  assert.equal(textSlot.model.id, 11);
  assert.equal(textSlot.unavailableReason, null);
  assert.deepEqual(textSlot.candidates.map((item) => item.model.id), [11]);

  assert.equal(videoSlot.model.id, 21);
  assert.equal(videoSlot.unavailableReason, "所属服务已停用");
  assert.deepEqual(videoSlot.candidates, []);
});

test("远程模型过滤同时匹配显示名、模型 ID 与归属方，并支持隐藏已添加与类型筛选", () => {
  const remote = [
    { id: "gemini-2.5-pro", displayName: "Gemini 2.5 旗舰版", ownedBy: "google", modelType: 1 },
    { id: "imagen-4", displayName: "Imagen 4", ownedBy: "google", modelType: 2 },
    { id: "veo-3", displayName: "Veo 3", ownedBy: "deepmind", modelType: 3 },
  ];
  const existing = new Set(["gemini-2.5-pro"]);

  const matchedByDisplayName = filterRemoteModels(remote, existing, {
    query: "旗舰版",
    modelType: 0,
    hideExisting: false,
  });
  assert.deepEqual(matchedByDisplayName.map((item) => item.id), ["gemini-2.5-pro"]);

  const hiddenExisting = filterRemoteModels(remote, existing, {
    query: "",
    modelType: 0,
    hideExisting: true,
  });
  assert.deepEqual(hiddenExisting.map((item) => item.id), ["imagen-4", "veo-3"]);

  const onlyVideo = filterRemoteModels(remote, existing, {
    query: "deepmind",
    modelType: 3,
    hideExisting: true,
  });
  assert.deepEqual(onlyVideo.map((item) => item.id), ["veo-3"]);
});
