import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import ts from "typescript";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const root = path.resolve(import.meta.dirname, "..");
const cache = new Map();

function load(relative) {
  const file = path.resolve(root, relative);
  if (cache.has(file)) {
    return cache.get(file).exports;
  }
  const loadedModule = { exports: {} };
  cache.set(file, loadedModule);
  const compiled = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      esModuleInterop: true,
    },
  }).outputText;
  const localRequire = (name) =>
    name.startsWith("@/")
      ? load(`${name.slice(2)}.ts`)
      : name.startsWith(".")
        ? load(`${path.resolve(path.dirname(file), name)}.ts`)
        : require(name);
  vm.runInThisContext(`(function(require,module,exports){${compiled}\n})`, {
    filename: file,
  })(localRequire, loadedModule, loadedModule.exports);
  return loadedModule.exports;
}

const pending = load("lib/generation-submission.ts");
const requests = load(
  "components/dashboard/generation/generation-submit-request.ts",
);
const labels = load("components/dashboard/generation/generation-utils.ts");
const forms = load("components/dashboard/generation/generation-types.ts");
const { resolveGenerationCapabilities, workflowInputIssue } = load(
  "lib/generation-capabilities.ts",
);
const videoRules = load("lib/video-model-rules.ts");

test("ComfyUI uses the server result instead of stale model flags and preset counts", () => {
  const model = { modelType: 2, comfyuiWorkflowId: 12, capabilityPresetCode: "old",
    config: JSON.stringify({ supportReferenceImages: false, maxReferenceImages: 2 }),
    generationConfig: { comfyuiWorkflowVersionId: 21, supportReferenceImages: true,
      minReferenceImages: 4, maxReferenceImages: 4, referenceImageInputFormats: ["url", "data_uri"] } };
  const capability = resolveGenerationCapabilities(model, [{ code: "old", modelType: 2,
    config: { supportReferenceImages: true, maxReferenceImages: 8 } }]);
  assert.equal(capability.supportsReferenceImages, true);
  assert.equal(capability.maxReferenceImages, 4);
  assert.equal(capability.minReferenceImages, 4);
  assert.equal(capability.unavailable, null);
});

test("ComfyUI does not substitute raw config when capability resolution fails", () => {
  const capability = resolveGenerationCapabilities({ modelType: 2, comfyuiWorkflowId: 12,
    config: '{"supportReferenceImages":true,"maxReferenceImages":8}', generationConfig: {},
    capabilityError: "使用上限与工作流冲突" }, []);
  assert.match(capability.unavailable, /冲突/);
  assert.equal(capability.supportsReferenceImages, false);
  assert.match(resolveGenerationCapabilities({ modelType: 2, comfyuiWorkflowId: 12 }, []).unavailable, /尚未加载/);
});

test("a workflow list without a declared upper bound is not converted to a one-image limit", () => {
  const capability = resolveGenerationCapabilities({ modelType: 2, comfyuiWorkflowId: 12,
    generationConfig: { comfyuiWorkflowVersionId: 21, supportReferenceImages: true,
      maxReferenceImages: null, maxImageInputs: null } }, []);
  assert.equal(labels.parseLimit(capability.maxReferenceImages, 1), Infinity);
  assert.equal(capability.maxImageInputs, Infinity);
});

test("workbench uses parsed workflow counts to reject missing, excess and unbound inputs", () => {
  const capability = resolveGenerationCapabilities({ modelType: 3, comfyuiWorkflowId: 12,
    generationConfig: { comfyuiWorkflowVersionId: 21, maxImageInputs: 4, comfyuiInputs: {
      referenceImages: { maximum: 4, minimum: 0, minimumWhenProvided: 4 },
    } } }, []);
  assert.equal(workflowInputIssue(capability, { referenceImages: 4 }), null);
  assert.equal(workflowInputIssue(capability, { referenceImages: 0 }), null);
  assert.match(workflowInputIssue(capability, { referenceImages: 3 }), /至少 4/);
  assert.match(workflowInputIssue(capability, { referenceImages: 5 }), /最多.*4/);
  assert.match(workflowInputIssue(capability, { firstFrame: 1 }), /不支持首帧/);
});
const officialPresets = ["volcengine", "minimax", "dashscope"].flatMap(
  (vendor) =>
    JSON.parse(
      fs.readFileSync(
        path.resolve(
          root,
          "../ai-fusion-video/src/main/resources/model-presets",
          vendor,
          "video.json",
        ),
        "utf8",
      ),
    ),
);
const officialCapabilities = (code) =>
  resolveGenerationCapabilities(
    {
      modelType: 3,
      capabilityPresetCode: code,
    },
    officialPresets,
  );
const scope = { userId: 42, teamId: null };
const original = {
  schemaVersion: 1,
  mode: "image",
  modelName: "固定模型",
  createdAt: "2026-10-13T00:00:00Z",
  request: {
    requestId: "original-request",
    scope,
    modelId: 7,
    prompt: "原始提示词",
    count: 2,
    refImageUrls: '["media://original-image"]',
  },
};

test("recovery preserves the original request, model, media and explicit account scope", () => {
  const restored = pending.readPendingGeneration(
    JSON.stringify(original),
    "image",
    scope,
  );
  assert.deepEqual(restored, original);
  assert.notEqual(
    pending.generationSubmissionKey("image", scope),
    pending.generationSubmissionKey("video", scope),
  );
  assert.notEqual(
    pending.generationSubmissionKey("image", scope),
    pending.generationSubmissionKey("image", {
      ...scope,
      teamId: 9,
    }),
  );
});

test("corrupt, foreign-scope or unsupported saved requests cannot silently turn into a new submission", () => {
  assert.throws(() => pending.readPendingGeneration("{", "image", scope));
  assert.throws(() =>
    pending.readPendingGeneration(
      JSON.stringify({ ...original, schemaVersion: 2 }),
      "image",
      scope,
    ),
  );
  assert.throws(() =>
    pending.readPendingGeneration(JSON.stringify(original), "video", scope),
  );
  assert.throws(() =>
    pending.readPendingGeneration(JSON.stringify(original), "image", {
      ...scope,
      teamId: 9,
    }),
  );
  assert.throws(() =>
    pending.readPendingGeneration(
      JSON.stringify({
        ...original,
        request: { ...original.request, requestId: "" },
      }),
      "image",
      scope,
    ),
  );
  assert.throws(() =>
    pending.readPendingGeneration(
      JSON.stringify({
        ...original,
        request: { ...original.request, count: 1000 },
      }),
      "image",
      scope,
    ),
  );
});

test("a late acknowledgement cannot remove a different pending request", () => {
  const key = pending.generationSubmissionKey("image", scope);
  const records = new Map([
    [
      key,
      JSON.stringify({
        ...original,
        request: { ...original.request, requestId: "new-request" },
      }),
    ],
  ]);
  const storage = {
    getItem: (key) => records.get(key) ?? null,
    removeItem: (key) => records.delete(key),
  };
  assert.throws(() =>
    pending.clearPendingGeneration(storage, key, "original-request"),
  );
  assert.equal(records.size, 1);
  pending.clearPendingGeneration(storage, key, "new-request");
  assert.equal(records.size, 0);
});

test("video submissions keep all reference classes and explicit false audio flags", () => {
  const input = requests.generationSubmitRequest(
    "video",
    {
      ...forms.EMPTY_GENERATION_FORM,
      prompt: " 完整的输入 ",
      firstFrame: ["media://first"],
      lastFrame: ["media://last"],
      referenceImages: ["media://image"],
      referenceVideos: ["media://video"],
      referenceAudios: ["media://audio"],
      seed: "0",
      generateAudio: false,
      watermark: false,
      cameraFixed: false,
      count: 3,
    },
    8,
  );
  assert.equal(input.modelId, 8);
  assert.equal(input.generateMode, "image2video");
  assert.equal(input.firstFrameImageUrl, "media://first");
  assert.equal(input.lastFrameImageUrl, "media://last");
  assert.deepEqual(JSON.parse(input.referenceAudioUrls), ["media://audio"]);
  assert.deepEqual(JSON.parse(input.referenceVideoUrls), ["media://video"]);
  assert.deepEqual(JSON.parse(input.referenceImageUrls), ["media://image"]);
  assert.equal(input.seed, 0);
  assert.equal(input.generateAudio, false);
  assert.equal(input.count, 3);
});

test("review, cancellation and partial completion never appear as successful or ordinary queued tasks", () => {
  const task = { status: 1, execution: { state: "UNKNOWN_SUBMISSION" } };
  assert.equal(labels.getTaskStatus(task).tone, "attention");
  assert.equal(
    labels.getTaskStatus({ ...task, execution: { state: "CANCEL_REQUESTED" } })
      .label,
    "取消中",
  );
  assert.equal(
    labels.getTaskStatus({ ...task, status: 4 }).label,
    "受理情况待核实",
  );
  assert.equal(
    labels.getTaskStatus({ status: 4, execution: { state: "CANCELLED" } })
      .label,
    "已取消",
  );
  assert.equal(
    labels.getTaskStatus({ status: 6, execution: { state: "PARTIAL_FAILED" } })
      .label,
    "部分完成",
  );
  assert.equal(
    labels.isGenerationTaskActive({
      status: 4,
      execution: { state: "CANCELLED" },
    }),
    false,
  );
  assert.equal(
    labels.isGenerationTaskActive({ status: 1, execution: null }),
    false,
  );
  assert.equal(
    labels.getTaskStatus({ status: 1, execution: null }).label,
    "历史状态待核实",
  );
});

test("H3 only offers concrete ratios for text and adaptive for frames without sending unsupported switches", () => {
  const cap = officialCapabilities("MiniMax-H3");
  assert.equal(
    videoRules
      .videoRatioOptions(
        cap.supportedAspectRatios,
        cap.videoRules,
        videoRules.videoInputState({}),
      )
      .includes("adaptive"),
    false,
  );
  assert.deepEqual(
    videoRules.videoRatioOptions(
      cap.supportedAspectRatios,
      cap.videoRules,
      videoRules.videoInputState({ firstFrame: ["media://frame"] }),
    ),
    ["adaptive"],
  );
  const form = labels.generationFormForModel(forms.EMPTY_GENERATION_FORM, cap);
  const request = requests.generationSubmitRequest(
    "video",
    { ...form, prompt: "海浪" },
    42,
    cap,
  );
  for (const key of ["generateAudio", "watermark", "seed", "cameraFixed"]) {
    assert.equal(request[key], undefined);
  }
  assert.match(
    videoRules.videoFormIssue(cap, { ...form, prompt: "海浪", duration: -1 }),
    /时长/,
  );
  assert.match(
    videoRules.videoFormIssue(cap, { ...form, prompt: "海浪", seed: "0" }),
    /种子/,
  );
});

test("media-only requests survive persistence and retain automatic duration", () => {
  const cap = officialCapabilities("doubao-seedance-2-5-260628");
  const form = {
    ...labels.generationFormForModel(forms.EMPTY_GENERATION_FORM, cap),
    referenceAudios: ["media://audio"],
    duration: -1,
  };
  assert.equal(videoRules.videoFormIssue(cap, form), null);
  const request = {
    ...requests.generationSubmitRequest("video", form, 7, cap),
    scope,
    requestId: "media-only",
  };
  const stored = { ...original, mode: "video", request };
  assert.equal(
    pending.readPendingGeneration(JSON.stringify(stored), "video", scope)
      .request.duration,
    -1,
  );
  assert.throws(() =>
    pending.readPendingGeneration(
      JSON.stringify({
        ...stored,
        request: { ...request, referenceAudioUrls: "[]" },
      }),
      "video",
      scope,
    ),
  );
  assert.equal(
    videoRules.videoPromptReady(
      "",
      cap.videoRules,
      videoRules.videoInputState({}),
    ),
    false,
  );
});

test("video presets initialize five-second requests while retaining discrete model limits", () => {
  const presetRoot = path.resolve(root, "../ai-fusion-video/src/main/resources/model-presets");
  const presets = fs.readdirSync(presetRoot, { recursive: true })
    .filter((file) => /^video(?:-[\w-]+)?\.json$/.test(path.basename(file)))
    .flatMap((file) => JSON.parse(fs.readFileSync(path.join(presetRoot, file), "utf8")));
  for (const preset of presets) {
    const cap = resolveGenerationCapabilities({ modelType: 3, capabilityPresetCode: preset.code }, presets);
    const form = labels.generationFormForModel({ ...forms.EMPTY_GENERATION_FORM, duration: 2 }, cap);
    const request = requests.generationSubmitRequest("video", { ...form, prompt: "海浪" }, 42, cap);
    const expected = ["sora-2", "sora-2-pro", "infistar:veo-3.1-fast-generate-preview"].includes(preset.code) ? 8 : 5;
    assert.equal(request.duration, expected, preset.code);
  }
  for (const code of ["infistar:wan3.0-video", "infistar:wan3.0-video-prime"]) {
    const cap = resolveGenerationCapabilities({ modelType: 3, capabilityPresetCode: code }, presets);
    const form = { ...labels.generationFormForModel(forms.EMPTY_GENERATION_FORM, cap), prompt: "海浪" };
    assert.equal(videoRules.videoFormIssue(cap, form), null);
    for (const duration of [2, 3, 4]) {
      assert.match(videoRules.videoFormIssue(cap, { ...form, duration }), /时长/, code);
    }
  }
  const configured = resolveGenerationCapabilities({
    modelType: 3,
    capabilityPresetCode: "infistar:wan3.0-video",
    config: '{"defaultDuration":10}',
  }, presets);
  assert.equal(labels.generationFormForModel(forms.EMPTY_GENERATION_FORM, configured).duration, 10);
});

test("Veo offers only documented seconds and updates duration when resolution changes", () => {
  const presets = JSON.parse(fs.readFileSync(path.resolve(root,
    "../ai-fusion-video/src/main/resources/model-presets/infistar/video.json"), "utf8"));
  const cap = resolveGenerationCapabilities({modelType: 3,
    capabilityPresetCode: "infistar:veo-3.1-fast-generate-preview"}, presets);
  const form = {...labels.generationFormForModel(forms.EMPTY_GENERATION_FORM, cap), prompt: "海浪"};
  assert.equal(form.duration, 8);
  assert.deepEqual(videoRules.videoDurationOptions(cap, "720p"), [4, 6, 8]);
  assert.deepEqual(videoRules.videoDurationOptions(cap, "1080p"), [8]);
  assert.equal(videoRules.videoDurationForResolution(6, cap, "1080p"), 8);
  assert.equal(videoRules.videoDurationForResolution(6, cap, "720p"), 6);
  assert.match(videoRules.videoFormIssue(cap, {...form, duration: 5}), /时长/);
  assert.match(videoRules.videoFormIssue(cap, {...form, resolution: "1080p", duration: 6}), /时长/);
  assert.equal(videoRules.videoFormIssue(cap, {...form, resolution: "720p", duration: 6}), null);
});

test("official input restrictions give actionable errors before submission", () => {
  const cap = officialCapabilities("wan3.0-video");
  const form = {
    ...labels.generationFormForModel(forms.EMPTY_GENERATION_FORM, cap),
    prompt: "海浪",
  };
  assert.match(
    videoRules.videoFormIssue(cap, {
      ...form,
      firstFrame: ["image"],
      referenceAudios: ["audio"],
    }),
    /不能混用/,
  );
  assert.match(
    videoRules.videoFormIssue(cap, { ...form, lastFrame: ["image"] }),
    /同时添加首帧/,
  );
  assert.match(
    videoRules.videoFormIssue(cap, { ...form, prompt: "字".repeat(20001) }),
    /20000/,
  );
});

test("Metaso resolution and mixed-reference restrictions are visible before submission", () => {
  const presets = JSON.parse(fs.readFileSync(path.resolve(root,
    "../ai-fusion-video/src/main/resources/model-presets/metaso/video.json"), "utf8"));
  const cap = resolveGenerationCapabilities({ modelType: 3, capabilityPresetCode: "metaso:MiniMax-H3" }, presets);
  const form = { ...labels.generationFormForModel(forms.EMPTY_GENERATION_FORM, cap), prompt: "海浪", duration: 30 };
  assert.equal(videoRules.videoFormIssue(cap, { ...form, resolution: "1080p" }), null);
  assert.equal(videoRules.videoMaxDuration(cap, "2K"), 15);
  assert.match(videoRules.videoFormIssue(cap, { ...form, resolution: "2K" }), /时长/);
  const mixed = { ...form, duration: 5, firstFrame: ["first"], referenceImages: ["character"], ratio: "adaptive" };
  assert.equal(videoRules.videoFormIssue(cap, mixed), null);
  assert.match(videoRules.videoFormIssue(cap, { ...mixed, referenceVideos: ["clip"] }), /不能与参考视频混用/);
});
