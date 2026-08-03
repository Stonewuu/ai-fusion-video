"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  parseConfigJson,
  isConfigRecord,
  normalizePlatform,
  isOpenAiReasoningPlatform,
  isAnthropicReasoningPlatform,
  isDashScopeReasoningPlatform,
  getPositiveNumberValue,
  getConfigBooleanValue,
  getConfigNumberValue,
  getConfigStringArray,
  SizesMap,
  SupportedSizesEditor,
  AspectRatiosEditor,
  getConfigFieldsByModelType,
  ToggleSettingCard,
  CapabilityNumberField,
} from "./model-config-support";

export function ModelConfigForm({
  modelType,
  platform,
  supportReasoning,
  configJson,
  onChange,
}: {
  modelType: number;
  platform?: string | null;
  supportReasoning: boolean;
  configJson: string | undefined;
  onChange: (json: string) => void;
}) {
  const [showRawJson, setShowRawJson] = useState(false);
  const [rawJsonDraft, setRawJsonDraft] = useState("");
  const [rawJsonError, setRawJsonError] = useState(false);

  const configObj = parseConfigJson(configJson);
  const normalizedPlatform = normalizePlatform(platform);
  const fields = getConfigFieldsByModelType(modelType, platform);
  const showReasoningConfig = modelType === 1 && supportReasoning;
  const includeReasoningMode = configObj.includeReasoning === true
    ? "true"
    : configObj.includeReasoning === false
      ? "false"
      : "auto";

  const emitChange = (next: Record<string, unknown>) => {
    const cleaned = Object.fromEntries(
      Object.entries(next).filter(([, v]) => v !== "" && v !== undefined && v !== null)
    );
    onChange(Object.keys(cleaned).length > 0 ? JSON.stringify(cleaned) : "");
  };

  const updateSimpleField = (key: string, raw: string) => {
    const next = { ...configObj };
    if (raw === "" || raw === undefined) {
      delete next[key];
    } else {
      const num = parseFloat(raw);
      next[key] = isNaN(num) ? raw : num;
    }
    emitChange(next);
  };

  const updateComplexField = (key: string, value: unknown) => {
    const next = { ...configObj };
    if (value === undefined || value === null) {
      delete next[key];
    } else {
      next[key] = value;
    }
    emitChange(next);
  };

  const updateSelectField = (key: string, rawValue: string) => {
    const next = { ...configObj };
    if (rawValue === "auto" || rawValue === "__unset__") {
      delete next[key];
    } else if (rawValue === "true" || rawValue === "false") {
      next[key] = rawValue === "true";
    } else {
      next[key] = rawValue;
    }
    emitChange(next);
  };

  const handleToggleRawJson = () => {
    if (!showRawJson) {
      setRawJsonDraft(Object.keys(configObj).length > 0 ? JSON.stringify(configObj, null, 2) : "");
      setRawJsonError(false);
    }
    setShowRawJson(!showRawJson);
  };

  const handleRawJsonApply = () => {
    if (!rawJsonDraft.trim()) {
      onChange("");
      setShowRawJson(false);
      return;
    }
    try {
      const parsed = JSON.parse(rawJsonDraft);
      if (!isConfigRecord(parsed)) {
        setRawJsonError(true);
        return;
      }
      emitChange(parsed);
      setRawJsonError(false);
      setShowRawJson(false);
    } catch {
      setRawJsonError(true);
    }
  };

  const supportsImageReferenceInputs = getConfigBooleanValue(configObj.supportReferenceImages);
  const supportsAsyncImageTaskMode = getConfigBooleanValue(configObj.asyncMode);
  const supportsFirstFrame = getConfigBooleanValue(configObj.supportFirstFrame);
  const supportsLastFrame = getConfigBooleanValue(configObj.supportLastFrame);
  const supportsVideoReferenceImages = getConfigBooleanValue(configObj.supportReferenceImages);
  const supportsReferenceVideos = getConfigBooleanValue(configObj.supportReferenceVideos);
  const supportsReferenceAudios = getConfigBooleanValue(configObj.supportReferenceAudios);
  const referenceImageInputFormats = getConfigStringArray(configObj.referenceImageInputFormats)
    .map(value => value.toLowerCase())
    .map(value => value === "base64" || value === "data-uri" ? "data_uri" : value)
    .filter(value => value === "url" || value === "data_uri");
  if (
    getConfigBooleanValue(configObj.supportDataUriInput) &&
    !referenceImageInputFormats.includes("data_uri")
  ) {
    referenceImageInputFormats.push("data_uri");
  }
  const supportsReferenceImageUrlInput = referenceImageInputFormats.includes("url");
  const supportsReferenceImageDataUriInput = referenceImageInputFormats.includes("data_uri");
  const supportsAnyVideoImageInput = supportsFirstFrame || supportsLastFrame || supportsVideoReferenceImages;

  const updateReferenceImageInputFormat = (format: "url" | "data_uri", enabled: boolean) => {
    const nextFormats = new Set(referenceImageInputFormats.filter(value => value === "url" || value === "data_uri"));
    if (enabled) nextFormats.add(format);
    else nextFormats.delete(format);
    emitChange({
      ...configObj,
      referenceImageInputFormats: Array.from(nextFormats),
      ...(format === "data_uri" ? { supportDataUriInput: enabled } : {}),
    });
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="text-xs text-muted-foreground">模型参数配置</Label>
        <button
          type="button"
          onClick={handleToggleRawJson}
          className="text-[10px] text-muted-foreground hover:text-foreground transition-colors px-1.5 py-0.5 rounded hover:bg-muted/50"
        >
          {showRawJson ? "切换表单" : "编辑 JSON"}
        </button>
      </div>

      {showRawJson ? (
        <div className="space-y-2">
          <textarea
            value={rawJsonDraft}
            onChange={e => {
              setRawJsonDraft(e.target.value);
              setRawJsonError(false);
            }}
            placeholder='{"temperature": 0.7, "maxTokens": 4096}'
            className={cn(
              "w-full min-h-[120px] rounded-lg border bg-background px-3 py-2 text-xs font-mono resize-y",
              "focus:outline-none focus:ring-2 focus:ring-ring/30",
              rawJsonError && "border-destructive focus:ring-destructive/30"
            )}
          />
          {rawJsonError && (
            <p className="text-[10px] text-destructive">JSON 格式错误，请检查语法</p>
          )}
          <div className="flex justify-end gap-1.5">
            <button
              type="button"
              onClick={() => setShowRawJson(false)}
              className="text-[10px] px-2 py-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
            >
              取消
            </button>
            <button
              type="button"
              onClick={handleRawJsonApply}
              className="text-[10px] px-2 py-1 rounded bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
            >
              应用
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {fields.length > 0 ? (
            <>
              {(() => {
                const simpleFields = fields.filter(f => f.type === "number" || f.type === "range");
                if (simpleFields.length === 0) return null;
                return (
                  <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
                    {simpleFields.map(field => {
                      const value = configObj[field.key];
                      const numValue = typeof value === "number" ? value : (typeof value === "string" ? parseFloat(value) : undefined);

                      if (field.type === "range") {
                        return (
                          <div key={field.key} className="space-y-1.5">
                            <div className="flex items-center justify-between">
                              <label className="text-[11px] text-muted-foreground">{field.label}</label>
                              <span className="text-[11px] font-mono text-foreground/80 tabular-nums min-w-[3ch] text-right">
                                {numValue !== undefined && !isNaN(numValue) ? numValue : "—"}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <input
                                type="range"
                                min={field.min ?? 0}
                                max={field.max ?? 1}
                                step={field.step ?? 0.1}
                                value={numValue !== undefined && !isNaN(numValue) ? numValue : field.defaultValue ?? field.min ?? 0}
                                onChange={e => updateSimpleField(field.key, e.target.value)}
                                className="flex-1 h-1.5 accent-primary cursor-pointer"
                              />
                              <button
                                type="button"
                                onClick={() => updateSimpleField(field.key, "")}
                                className="text-[9px] text-muted-foreground hover:text-destructive transition-colors shrink-0"
                                title="清除"
                              >
                                ✕
                              </button>
                            </div>
                            {field.hint && <p className="text-[10px] text-muted-foreground/70">{field.hint}</p>}
                          </div>
                        );
                      }

                      return (
                        <div key={field.key} className="space-y-1">
                          <label className="text-[11px] text-muted-foreground">{field.label}</label>
                          <Input
                            type="number"
                            min={field.min}
                            max={field.max}
                            step={field.step}
                            placeholder={field.placeholder}
                            value={numValue !== undefined && !isNaN(numValue) ? numValue : ""}
                            onChange={e => updateSimpleField(field.key, e.target.value)}
                            className="text-xs font-mono h-8"
                          />
                          {field.hint && <p className="text-[10px] text-muted-foreground/70">{field.hint}</p>}
                        </div>
                      );
                    })}
                  </div>
                );
              })()}

              {fields.some(f => f.type === "supported-sizes") && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3">
                  <SupportedSizesEditor
                    value={configObj.supportedSizes as SizesMap | undefined}
                    onChange={v => updateComplexField("supportedSizes", v)}
                  />
                </div>
              )}

              {fields
                .filter(f => f.type === "aspect-ratios")
                .map(field => (
                  <div key={field.key} className="rounded-lg border border-border/40 bg-muted/20 p-3">
                    <AspectRatiosEditor
                      label={field.label}
                      hint={field.hint}
                      presetOptions={field.presetOptions}
                      value={configObj[field.key] as string[] | undefined}
                      onChange={v => updateComplexField(field.key, v)}
                    />
                  </div>
                ))}

              {modelType === 2 && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
                  <div>
                    <Label className="text-[11px] text-muted-foreground">参考图能力</Label>
                    <p className="text-[10px] text-muted-foreground/70 mt-1">控制 generate_image 是否允许传 imageUrls，以及允许的参考图数量。</p>
                  </div>

                  <ToggleSettingCard
                    checked={supportsImageReferenceInputs}
                    title="支持参考图输入"
                    description="开启后，agent 才会把 imageUrls 传给当前图片模型。关闭时会要求只走文生图。"
                    onToggle={() => {
                      const nextEnabled = !supportsImageReferenceInputs;
                      emitChange({
                        ...configObj,
                        supportReferenceImages: nextEnabled,
                        minReferenceImages: nextEnabled ? configObj.minReferenceImages : 0,
                        maxReferenceImages: nextEnabled ? configObj.maxReferenceImages : 0,
                        referenceImageInputFormats: nextEnabled ? configObj.referenceImageInputFormats : [],
                        supportDataUriInput: nextEnabled ? configObj.supportDataUriInput : false,
                      });
                    }}
                  />

                  <div className="grid gap-2.5 sm:grid-cols-2">
                    <ToggleSettingCard
                      checked={supportsReferenceImageUrlInput}
                      title="允许 URL 传递"
                      description="有公网对象存储或后端资源公网地址时，参考图直接以 URL 传给上游。"
                      disabled={!supportsImageReferenceInputs}
                      onToggle={() => updateReferenceImageInputFormat("url", !supportsReferenceImageUrlInput)}
                    />
                    <ToggleSettingCard
                      checked={supportsReferenceImageDataUriInput}
                      title="允许 base64 / Data URI"
                      description="没有公网访问地址时，服务端读取图片并转换为 Data URI。"
                      disabled={!supportsImageReferenceInputs}
                      onToggle={() => updateReferenceImageInputFormat("data_uri", !supportsReferenceImageDataUriInput)}
                    />
                  </div>

                  <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
                    <CapabilityNumberField
                      label="最少参考图数量"
                      value={getConfigNumberValue(configObj.minReferenceImages)}
                      onChange={value => updateSimpleField("minReferenceImages", value)}
                      min={0}
                      step={1}
                      disabled={!supportsImageReferenceInputs}
                      placeholder="例如：0"
                      hint="通常填 0；只有模型明确要求至少上传多张参考图时才需要设置。"
                    />
                    <CapabilityNumberField
                      label="最多参考图数量"
                      value={getConfigNumberValue(configObj.maxReferenceImages)}
                      onChange={value => updateSimpleField("maxReferenceImages", value)}
                      min={0}
                      step={1}
                      disabled={!supportsImageReferenceInputs}
                      placeholder="例如：3"
                      hint="限制单次 generate_image 可传入的 imageUrls 数量。"
                    />
                  </div>
                </div>
              )}

              {modelType === 2 && normalizedPlatform === "openai_compatible" && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
                  <div>
                    <Label className="text-[11px] text-muted-foreground">OpenAI 兼容图片任务模式</Label>
                    <p className="text-[10px] text-muted-foreground/70 mt-1">
                      开启后，图片生成会按异步任务模式提交，并通过通用任务查询接口轮询结果。
                    </p>
                  </div>

                  <ToggleSettingCard
                    checked={supportsAsyncImageTaskMode}
                    title="启用异步模式"
                    description="适用于返回 task_id 后再查询结果的 OpenAI 兼容图片接口；关闭时保持同步 images/generations 或 images/edits 调用。"
                    onToggle={() => updateComplexField("asyncMode", !supportsAsyncImageTaskMode)}
                  />
                </div>
              )}

              {modelType === 3 && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
                  <div>
                    <Label className="text-[11px] text-muted-foreground">多模态输入能力</Label>
                    <p className="text-[10px] text-muted-foreground/70 mt-1">控制 generate_video 是否允许首帧、尾帧、参考图、参考视频和参考音频，以及对应数量上限。</p>
                  </div>

                  <div className="grid gap-2.5 [grid-template-columns:repeat(auto-fit,minmax(240px,1fr))]">
                    <ToggleSettingCard
                      checked={supportsFirstFrame}
                      title="支持首帧图"
                      description="允许传 firstFrameImageUrl 来锁定开场画面。"
                      onToggle={() => updateComplexField("supportFirstFrame", !supportsFirstFrame)}
                    />
                    <ToggleSettingCard
                      checked={supportsLastFrame}
                      title="支持尾帧图"
                      description="允许传 lastFrameImageUrl 来约束结尾画面。"
                      onToggle={() => updateComplexField("supportLastFrame", !supportsLastFrame)}
                    />
                    <ToggleSettingCard
                      checked={supportsVideoReferenceImages}
                      title="支持参考图"
                      description="允许传 referenceImageUrls；适合角色、场景或多图参考。"
                      onToggle={() => {
                        const nextEnabled = !supportsVideoReferenceImages;
                        emitChange({
                          ...configObj,
                          supportReferenceImages: nextEnabled,
                          maxReferenceImages: nextEnabled ? configObj.maxReferenceImages : 0,
                        });
                      }}
                    />
                    <ToggleSettingCard
                      checked={supportsReferenceVideos}
                      title="支持参考视频"
                      description="允许传 referenceVideoUrls，用于动作或镜头风格参考。"
                      onToggle={() => {
                        const nextEnabled = !supportsReferenceVideos;
                        emitChange({
                          ...configObj,
                          supportReferenceVideos: nextEnabled,
                          maxReferenceVideos: nextEnabled ? configObj.maxReferenceVideos : 0,
                        });
                      }}
                    />
                    <ToggleSettingCard
                      checked={supportsReferenceAudios}
                      title="支持参考音频"
                      description="允许传 referenceAudioUrls，用于节奏或音频条件参考。"
                      onToggle={() => {
                        const nextEnabled = !supportsReferenceAudios;
                        emitChange({
                          ...configObj,
                          supportReferenceAudios: nextEnabled,
                          maxReferenceAudios: nextEnabled ? configObj.maxReferenceAudios : 0,
                        });
                      }}
                    />
                  </div>

                  <div className="space-y-2">
                    <div>
                      <Label className="text-[11px] text-muted-foreground">图片传递模式</Label>
                      <p className="mt-1 text-[10px] text-muted-foreground/70">
                        同时作用于首帧图、尾帧图和 referenceImageUrls；至少启用一种才能提交图片输入。
                      </p>
                    </div>
                    <div className="grid gap-2.5 sm:grid-cols-2">
                      <ToggleSettingCard
                        checked={supportsReferenceImageUrlInput}
                        title="允许 URL 传递"
                        description="有公网对象存储或后端资源公网地址时，直接传递图片 URL。"
                        disabled={!supportsAnyVideoImageInput}
                        onToggle={() => updateReferenceImageInputFormat("url", !supportsReferenceImageUrlInput)}
                      />
                      <ToggleSettingCard
                        checked={supportsReferenceImageDataUriInput}
                        title="允许 base64 / Data URI"
                        description="没有公网访问地址时，将图片转换为 Data URI 后再提交。"
                        disabled={!supportsAnyVideoImageInput}
                        onToggle={() => updateReferenceImageInputFormat("data_uri", !supportsReferenceImageDataUriInput)}
                      />
                    </div>
                  </div>

                  <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
                    <CapabilityNumberField
                      label="最少图片输入数"
                      value={getConfigNumberValue(configObj.minImageInputs)}
                      onChange={value => updateSimpleField("minImageInputs", value)}
                      min={0}
                      step={1}
                      placeholder="例如：1"
                      hint="计数包含 firstFrameImageUrl、lastFrameImageUrl 和 referenceImageUrls。"
                    />
                    <CapabilityNumberField
                      label="最多图片输入数"
                      value={getConfigNumberValue(configObj.maxImageInputs)}
                      onChange={value => updateSimpleField("maxImageInputs", value)}
                      min={0}
                      step={1}
                      placeholder="例如：3"
                      hint="用于限制图片类输入总数，避免首尾帧与参考图一起超限。"
                    />
                    <CapabilityNumberField
                      label="最多参考图数量"
                      value={getConfigNumberValue(configObj.maxReferenceImages)}
                      onChange={value => updateSimpleField("maxReferenceImages", value)}
                      min={0}
                      step={1}
                      disabled={!supportsVideoReferenceImages}
                      placeholder="例如：3"
                      hint="referenceImageUrls 的单独上限。"
                    />
                    <CapabilityNumberField
                      label="最多参考视频数量"
                      value={getConfigNumberValue(configObj.maxReferenceVideos)}
                      onChange={value => updateSimpleField("maxReferenceVideos", value)}
                      min={0}
                      step={1}
                      disabled={!supportsReferenceVideos}
                      placeholder="例如：1"
                      hint="referenceVideoUrls 的单独上限。"
                    />
                    <CapabilityNumberField
                      label="最多参考音频数量"
                      value={getConfigNumberValue(configObj.maxReferenceAudios)}
                      onChange={value => updateSimpleField("maxReferenceAudios", value)}
                      min={0}
                      step={1}
                      disabled={!supportsReferenceAudios}
                      placeholder="例如：1"
                      hint="referenceAudioUrls 的单独上限。"
                    />
                  </div>
                </div>
              )}

              {showReasoningConfig && isOpenAiReasoningPlatform(normalizedPlatform) && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-3">
                  <div>
                    <Label className="text-[11px] text-muted-foreground">思考内容返回</Label>
                    <p className="text-[10px] text-muted-foreground/70 mt-1">自动模式下，只要开启“支持思考”或填写推理参数，后端会自动请求 reasoning 内容。</p>
                  </div>

                  <div className="space-y-1.5">
                    <Select
                      value={includeReasoningMode}
                      onValueChange={v => updateSelectField("includeReasoning", String(v))}
                      items={[
                        { value: "auto", label: "自动" },
                        { value: "true", label: "显式返回思考内容" },
                        { value: "false", label: "显式关闭思考内容返回" },
                      ]}
                    >
                      <SelectTrigger className="w-full text-sm">
                        <SelectValue placeholder="选择思考内容返回策略" />
                      </SelectTrigger>
                      <SelectContent className="text-sm">
                        <SelectGroup>
                          <SelectItem value="auto" className="text-sm">自动</SelectItem>
                          <SelectItem value="true" className="text-sm">显式返回思考内容</SelectItem>
                          <SelectItem value="false" className="text-sm">显式关闭思考内容返回</SelectItem>
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1.5">
                      <Label className="text-[11px] text-muted-foreground">Thinking Budget</Label>
                      <Input
                        type="number"
                        min={0}
                        step={1}
                        placeholder="例如：2048"
                        value={getPositiveNumberValue(typeof configObj.thinkingBudget === "number" ? configObj.thinkingBudget : undefined)}
                        onChange={e => updateSimpleField("thinkingBudget", e.target.value)}
                        className="text-xs font-mono h-8"
                      />
                      <p className="text-[10px] text-muted-foreground/70">面向支持 reasoning budget 的 OpenAI 兼容接入。</p>
                  </div>
                </div>
              )}

              {showReasoningConfig && isAnthropicReasoningPlatform(normalizedPlatform) && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3 space-y-1.5">
                  <Label className="text-[11px] text-muted-foreground">Thinking Budget</Label>
                  <Input
                    type="number"
                    min={0}
                    step={1}
                    placeholder="例如：1024"
                    value={getPositiveNumberValue(typeof configObj.thinkingBudget === "number" ? configObj.thinkingBudget : undefined)}
                    onChange={e => updateSimpleField("thinkingBudget", e.target.value)}
                    className="text-xs font-mono h-8"
                  />
                  <p className="text-[10px] text-muted-foreground/70">Claude 开启“支持思考”后，未填写时后端默认使用 1024。更细粒度的 thinking 结构仍可通过 JSON 编辑器覆盖。</p>
                </div>
              )}

              {showReasoningConfig && isDashScopeReasoningPlatform(normalizedPlatform) && (
                <div className="rounded-lg border border-border/40 bg-muted/20 p-3">
                  <p className="text-[10px] text-muted-foreground/70">DashScope 的思考模式由下方“支持思考”能力开关控制；如果后续需要 provider 专属参数，仍可使用 JSON 编辑器补充。</p>
                </div>
              )}
            </>
          ) : (
            <p className="text-[10px] text-muted-foreground/60 italic">当前模型类型无预定义配置项，可点击「编辑 JSON」手动配置</p>
          )}
        </div>
      )}
    </div>
  );
}
