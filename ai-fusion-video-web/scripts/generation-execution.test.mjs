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
const { resolutionChoices } = load("lib/image-output-sizes.ts");
const imageOutputs = load("lib/image-output-sizes.ts");
const canvasImages = load("lib/canvas/image-output.ts");
const executionDetails = load("components/dashboard/generation/generation-execution-presentation.ts");
const references = load("components/dashboard/generation/generation-reference-input.ts");
const { generationDurationScale } = load("components/dashboard/generation/generation-duration-scale.ts");

test("连续秒数列表与范围时长都通过滑条选择实际秒数", () => {
  const enumerated = generationDurationScale(2, 30, Array.from({ length: 29 }, (_, index) => index + 2));
  assert.equal(enumerated.seconds(0), 2);
  assert.equal(enumerated.seconds(enumerated.max), 30);
  assert.equal(enumerated.seconds(enumerated.position(20)), 20);
  const range = generationDurationScale(4, 30);
  assert.equal(range.seconds(range.position(20)), 20);
  assert.equal(range.min, 4);
  assert.equal(range.max, 30);
});

test("离散时长滑条只选择模型支持的秒数，并随分辨率更新", () => {
  const cap = { supportedDurations: [4, 6, 8], supportedDurationsByResolution: { "4K": [8] }, minDuration: 4, maxDuration: 8 };
  const scale = generationDurationScale(4, 8, videoRules.videoDurationOptions(cap, "720p"));
  assert.deepEqual(Array.from({ length: scale.max + 1 }, (_, index) => scale.seconds(index)), [4, 6, 8]);
  assert.equal(scale.seconds(scale.position(4) + 1), 6);
  const highResolution = generationDurationScale(4, 8, videoRules.videoDurationOptions(cap, "4K"));
  assert.equal(highResolution.min, highResolution.max);
  assert.equal(highResolution.seconds(0), 8);
});

const imageReference = "media://11111111-1111-4111-8111-111111111111";
const videoReference = "media://22222222-2222-4222-8222-222222222222";
const audioReference = "media://33333333-3333-4333-8333-333333333333";
const referenceCapabilities = {
  supportsFirstFrame: true, supportsLastFrame: true, supportsReferenceImages: true,
  supportsReferenceVideos: true, supportsReferenceAudios: true,
  maxImageInputs: 5, maxReferenceImages: 3, maxReferenceVideos: 2, maxReferenceAudios: 2,
};

test("从视频结果添加引用只传原视频，不用封面替代", () => {
  const form = { ...forms.EMPTY_GENERATION_FORM };
  const item = { videoUrl: videoReference, coverUrl: imageReference };
  const added = references.appendResultReference("video", form, referenceCapabilities, item);
  assert.deepEqual(added.form.referenceVideos, [videoReference]);
  assert.deepEqual(added.form.referenceImages, []);
  const request = requests.generationSubmitRequest("video", added.form, 7);
  assert.equal(request.referenceVideoUrls, JSON.stringify([videoReference]));
  assert.equal(request.referenceImageUrls, undefined);
  assert.deepEqual(form.referenceVideos, []);
});

test("模型不支持视频引用、结果未就绪或数量已满时明确拒绝添加", () => {
  const item = { videoUrl: videoReference, coverUrl: imageReference };
  assert.throws(() => references.appendResultReference("video", forms.EMPTY_GENERATION_FORM,
    { ...referenceCapabilities, supportsReferenceVideos: false }, item), /不支持参考视频/);
  assert.throws(() => references.appendResultReference("video", forms.EMPTY_GENERATION_FORM,
    referenceCapabilities, { ...item, videoUrl: null }), /尚未就绪/);
  assert.throws(() => references.appendResultReference("video",
    { ...forms.EMPTY_GENERATION_FORM, referenceVideos: [videoReference] }, referenceCapabilities, item), /已在/);
  assert.throws(() => references.appendResultReference("video", forms.EMPTY_GENERATION_FORM,
    { ...referenceCapabilities, maxReferenceVideos: 0 }, item), /数量已达上限/);
});

test("再次使用恢复全部参考素材和关闭参数，提交时保留原始媒体引用", () => {
  const task = { prompt: "按原视频的运动生成", ratio: "16:9", resolution: "480P", count: 2,
    duration: 4, seed: 0, watermark: false, generateAudio: false, cameraFixed: false,
    firstFrameImageUrl: imageReference, lastFrameImageUrl: imageReference,
    referenceImageUrls: JSON.stringify([imageReference]),
    referenceVideoUrls: JSON.stringify([videoReference]), referenceAudioUrls: JSON.stringify([audioReference]) };
  const restored = references.generationFormFromTask("video", task,
    { ...forms.EMPTY_GENERATION_FORM, watermark: true, generateAudio: true });
  const request = requests.generationSubmitRequest("video", restored, 7);
  assert.equal(request.firstFrameImageUrl, imageReference);
  assert.equal(request.lastFrameImageUrl, imageReference);
  assert.equal(request.referenceImageUrls, task.referenceImageUrls);
  assert.equal(request.referenceVideoUrls, task.referenceVideoUrls);
  assert.equal(request.referenceAudioUrls, task.referenceAudioUrls);
  assert.equal(request.prompt, task.prompt);
  assert.equal(request.duration, 4);
  assert.equal(request.seed, 0);
  assert.equal(request.generateAudio, false);
  assert.equal(request.watermark, false);
  assert.equal(references.taskReferences("video", task).filter(item => item.mediaType === "video")[0].url, videoReference);
});

test("再次使用无引用的任务会清除当前草稿引用，图片任务保留自己的参考图", () => {
  const defaults = { ...forms.EMPTY_GENERATION_FORM, referenceVideos: [videoReference], referenceImages: [imageReference] };
  const restored = references.generationFormFromTask("video", { prompt: "新的风景", count: 1 }, defaults);
  assert.deepEqual(restored.referenceVideos, []);
  assert.deepEqual(restored.referenceImages, []);
  assert.equal(restored.seed, "");
  const image = references.generationFormFromTask("image",
    { prompt: "保持人物", count: 1, refImageUrls: JSON.stringify([imageReference]) }, defaults);
  assert.deepEqual(image.referenceImages, [imageReference]);
  assert.deepEqual(image.referenceVideos, []);
});

test("移除首帧不会同时移除使用同一图片的尾帧", () => {
  const form = { ...forms.EMPTY_GENERATION_FORM, firstFrame: [imageReference], lastFrame: [imageReference] };
  const removed = references.removeGenerationReference(form,
    { url: imageReference, kind: "firstFrame", mediaType: "image", label: "首帧" });
  assert.deepEqual(removed.firstFrame, []);
  assert.deepEqual(removed.lastFrame, [imageReference]);
  assert.deepEqual(form.firstFrame, [imageReference]);
});

test("引用记录损坏时不把再次使用变成没有参考素材的生成", () => {
  assert.throws(() => references.generationFormFromTask("video",
    { prompt: "保持原主体", count: 1, referenceVideoUrls: "invalid-json" }, forms.EMPTY_GENERATION_FORM), /无法读取/);
});

test("本地结束但供应商受理未知时，详情仍显示受理待确认", () => {
  const item = { state: "UNKNOWN_SUBMISSION", providerTaskId: null, availableActions: [] };
  assert.equal(executionDetails.executionStatus({ state: "FAILED", items: [item] }), "受理待确认");
  assert.match(executionDetails.executionGuidance(item), /供应商平台核对/);
  assert.equal(executionDetails.executionStatus({ state: "FAILED", items: [{ state: "FAILED" }] }), "生成失败");
  assert.equal(executionDetails.executionTime(null), "—");
});

test("执行参数保留零种子和关闭状态，不把未配置参数显示成默认值", () => {
  const values = executionDetails.executionParameters({ kind: "VIDEO", input: { parameters: {
    count: 2, duration: -1, seed: 0, generateAudio: false, ratio: null, resolution: null,
  } } });
  const fields = Object.fromEntries(values.map(({ label, value }) => [label, value]));
  assert.deepEqual(fields, { 数量: "2 段", 画面比例: "—", 分辨率: "—", 时长: "自动", 声音: "关闭", 随机种子: "0" });
});

test("已收到结果但保存失败时，详情引导保存结果而不是重新生成", () => {
  const guidance = executionDetails.executionGuidance({ state: "FAILED", availableActions: ["RETRY_INGEST"] });
  assert.match(guidance, /重新保存结果/);
  assert.doesNotMatch(guidance, /重新生成/);
});

test("同步返回结果但没有任务编号时，不误报供应商尚未响应", () => {
  assert.equal(executionDetails.executionProviderTask({ providerTaskId: null, attempts: [{ submitted: true, acknowledged: true }] }), "—");
  assert.equal(executionDetails.executionProviderTask({ providerTaskId: null, attempts: [{ submitted: true, acknowledged: false }] }), "未返回");
  assert.equal(executionDetails.executionProviderTask({ providerTaskId: "provider-1", attempts: [] }), "provider-1");
});

test("画布与工作台使用同一比例档位选择，切换模型清除旧的自定义宽高", () => {
  const parameters = {
    supportedAspectRatios: ["1:1", "9:16"], supportedResolutions: ["2K", "4K", "auto"],
    supportAutoSize: true, defaultAspectRatio: "9:16", defaultResolution: "2K",
    supportedSizes: { "2K": { "9:16": "1152x2048" }, "4K": { "1:1": "2880x2880" } },
  };
  const model = { id: 7, parameters };
  const saved = canvasImages.canvasImageDefaults({ modelId: 7, width: 1920, height: 1080, prompt: "海边" }, model);
  assert.deepEqual(saved, { modelId: 7, prompt: "海边", ratio: "9:16", resolution: "2K" });
  const options = imageOutputs.imageOutputCapabilities(parameters);
  assert.deepEqual(imageOutputs.changeImageOutput(options, saved, { ratio: "1:1" }),
    { ...saved, ratio: "1:1", resolution: "4K" });
  assert.match(canvasImages.canvasImageOutputIssue({ ratio: "9:16", resolution: "4K" }, model), /分辨率/);
  assert.equal(canvasImages.canvasImageOutputIssue(saved, model), null);
});

test("画布保留按像素提交的协议，并能将已保存参数显示为档位", () => {
  const options = imageOutputs.imageOutputCapabilities({
    supportedAspectRatios: ["1:1", "3:4"],
    supportedResolutions: ["1024x1024", "768x1024"],
    supportedSizes: { standard: { "1:1": "1024x1024", "3:4": "768x1024" } },
  });
  assert.deepEqual(imageOutputs.imageOutputRequest(options, { ratio: "3:4", resolution: "standard" }),
    { ratio: "3:4", resolution: "768x1024" });
  assert.deepEqual(imageOutputs.imageOutputSelection(options, { ratio: "3:4", resolution: "768x1024" }),
    { ratio: "3:4", resolution: "standard" });
  assert.match(imageOutputs.imageOutputIssue(options, { ratio: "1:1", resolution: "768x1024" }), /分辨率/);
});

test("自定义尺寸检查像素倍数、范围与冲突，自动尺寸不会保留固定宽高", () => {
  const options = imageOutputs.imageOutputCapabilities({
    supportedAspectRatios: ["auto", "1:1", "16:9"], supportedResolutions: ["auto", "2K"],
    supportAutoSize: true, supportCustomSize: true, sizeMultiple: 16,
    minPixels: 655360, maxPixels: 8294400, maxEdge: 3840, maxAspectRatio: 3,
    supportedSizes: { "2K": { "1:1": "2048x2048", "16:9": "2048x1152" } },
  });
  assert.match(imageOutputs.imageOutputIssue(options, { width: 1920, height: 1080 }), /16 的倍数/);
  assert.match(imageOutputs.imageOutputIssue(options, { width: 512, height: 512 }), /总像素/);
  assert.match(imageOutputs.imageOutputIssue(options, { ratio: "auto", resolution: "auto", width: 1024, height: 1024 }), /不能同时/);
  assert.match(imageOutputs.imageOutputIssue(options, { ratio: "16:9", resolution: "2K", width: 1024, height: 1024 }), /不一致/);
  assert.equal(imageOutputs.imageOutputIssue(options, { ratio: "16:9", resolution: "2K" }), null);
  assert.equal(imageOutputs.imageOutputIssue(options, { width: 1920, height: 1088 }), null);
});

test("GPT 图片模型的每个固定比例都能选择三个档位，提交时保留档位和比例", () => {
  const presets = ["openai/image.json", "openai/image-25.json"].flatMap((file) =>
    JSON.parse(fs.readFileSync(path.resolve(root, "../ai-fusion-video/src/main/resources/model-presets", file), "utf8")),
  );
  const ratios = ["1:1", "2:3", "3:2", "4:3", "3:4", "5:4", "4:5", "16:9", "9:16", "2:1", "1:2", "21:9", "9:21"];
  for (const code of ["gpt-image-2", "gpt-image-2.5-sunburst", "gpt-image-2.5-flare"]) {
    const capability = resolveGenerationCapabilities({ modelType: 2, capabilityPresetCode: code }, presets);
    const form = labels.generationFormForModel(forms.EMPTY_GENERATION_FORM, capability);
    assert.deepEqual([form.ratio, form.resolution], ["1:1", "1K"]);
    assert.deepEqual(new Set(capability.supportedAspectRatios), new Set([...ratios, "auto"]));
    for (const ratio of ratios) {
      assert.deepEqual(resolutionChoices(capability, ratio), ["1K", "2K", "4K"]);
    }
    assert.deepEqual(resolutionChoices(capability, "auto"), ["auto"]);
    const request = requests.generationSubmitRequest("image", { ...form, ratio: "9:16", resolution: "2K" }, 15, capability);
    assert.deepEqual([request.ratio, request.resolution], ["9:16", "2K"]);
    assert.equal(capability.imageSizes["2K"]["9:16"], "1152x2048");
  }
});

test("按模型声明的格式提交图片尺寸，切换比例时只显示可用档位", () => {
  const presets = JSON.parse(fs.readFileSync(path.resolve(root, "../ai-fusion-video/src/main/resources/model-presets/agnes/image.json"), "utf8"));
  const exact = resolveGenerationCapabilities({ modelType: 2, capabilityPresetCode: "agnes-image-2.0-flash" }, presets);
  const form = { ...forms.EMPTY_GENERATION_FORM, ratio: "3:4", resolution: "standard" };
  assert.deepEqual(resolutionChoices(exact, "3:4"), ["standard"]);
  assert.equal(requests.generationSubmitRequest("image", form, 7, exact).resolution, "768x1024");

  const config = { supportedAspectRatios: ["9:16", "1:1"], supportedResolutions: ["2K", "4K"] };
  const native = resolveGenerationCapabilities({ modelType: 2, generationConfig: config }, []);
  const portrait = { ...form, ratio: "9:16", resolution: "2K" };
  const request = requests.generationSubmitRequest("image", portrait, 7, native);
  assert.deepEqual([request.resolution, request.ratio], ["2K", "9:16"]);
  const limited = resolveGenerationCapabilities({ modelType: 2, generationConfig: { ...config,
    supportedSizes: { "2K": { "9:16": "1152x2048" }, "4K": { "1:1": "2880x2880" } } } }, []);
  assert.deepEqual(resolutionChoices(limited, "9:16"), ["2K"]);
  assert.deepEqual(resolutionChoices(limited, "1:1"), ["4K"]);
});

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
    "远端结果未知",
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

test("视频预设使用各自的默认时长，并保留模型的离散时长限制", () => {
  const presetRoot = path.resolve(root, "../ai-fusion-video/src/main/resources/model-presets");
  const presets = fs.readdirSync(presetRoot, { recursive: true })
    .filter((file) => /^video(?:-[\w-]+)?\.json$/.test(path.basename(file)))
    .flatMap((file) => JSON.parse(fs.readFileSync(path.join(presetRoot, file), "utf8")));
  const eightSecondPresets = new Set([
    "sora-2",
    "sora-2-pro",
    "infistar:veo-3.1-fast-generate-preview",
    "google:veo-3.1-generate-001",
    "google:veo-3.1-fast-generate-001",
    "google:veo-3.1-lite-generate-001",
  ]);
  for (const preset of presets) {
    const cap = resolveGenerationCapabilities({ modelType: 3, capabilityPresetCode: preset.code }, presets);
    const form = labels.generationFormForModel({ ...forms.EMPTY_GENERATION_FORM, duration: 2 }, cap);
    const request = requests.generationSubmitRequest("video", { ...form, prompt: "海浪" }, 42, cap);
    const expected = eightSecondPresets.has(preset.code) ? 8 : 5;
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
