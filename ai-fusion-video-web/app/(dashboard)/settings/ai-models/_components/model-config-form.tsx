"use client";

import { useId } from "react";
import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AspectRatiosEditor,
  CapabilityNumberField,
  getConfigBooleanValue,
  getConfigFieldsByModelType,
  getConfigNumberValue,
  getConfigStringArray,
  getPositiveNumberValue,
  isAnthropicReasoningPlatform,
  isDashScopeReasoningPlatform,
  isOpenAiReasoningPlatform,
  normalizePlatform,
  parseConfigJson,
  SupportedSizesEditor,
  ToggleSettingCard,
  type SizesMap,
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
  const idPrefix = useId();
  const configObj = parseConfigJson(configJson);
  const normalizedPlatform = normalizePlatform(platform);
  const fields = getConfigFieldsByModelType(modelType, platform);
  const showReasoningConfig = modelType === 1 && supportReasoning;
  const includeReasoningMode =
    configObj.includeReasoning === true
      ? "true"
      : configObj.includeReasoning === false
        ? "false"
        : "auto";

  const emitChange = (next: Record<string, unknown>) => {
    const cleaned = Object.fromEntries(
      Object.entries(next).filter(
        ([, value]) => value !== "" && value !== undefined && value !== null,
      ),
    );
    onChange(Object.keys(cleaned).length > 0 ? JSON.stringify(cleaned) : "");
  };

  const updateSimpleField = (key: string, raw: string) => {
    const next = { ...configObj };
    if (raw === "" || raw === undefined) {
      delete next[key];
    } else {
      const num = parseFloat(raw);
      next[key] = Number.isNaN(num) ? raw : num;
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

  const supportsImageReferenceInputs = getConfigBooleanValue(
    configObj.supportReferenceImages,
  );
  const supportsAsyncImageTaskMode = getConfigBooleanValue(configObj.asyncMode);
  const supportsFirstFrame = getConfigBooleanValue(configObj.supportFirstFrame);
  const supportsLastFrame = getConfigBooleanValue(configObj.supportLastFrame);
  const supportsVideoReferenceImages = getConfigBooleanValue(
    configObj.supportReferenceImages,
  );
  const supportsReferenceVideos = getConfigBooleanValue(
    configObj.supportReferenceVideos,
  );
  const supportsReferenceAudios = getConfigBooleanValue(
    configObj.supportReferenceAudios,
  );
  const referenceImageInputFormats = getConfigStringArray(
    configObj.referenceImageInputFormats,
  )
    .map((value) => value.toLowerCase())
    .map((value) => (value === "base64" || value === "data-uri" ? "data_uri" : value))
    .filter((value) => value === "url" || value === "data_uri");
  if (
    getConfigBooleanValue(configObj.supportDataUriInput)
    && !referenceImageInputFormats.includes("data_uri")
  ) {
    referenceImageInputFormats.push("data_uri");
  }
  const supportsReferenceImageUrlInput = referenceImageInputFormats.includes("url");
  const supportsReferenceImageDataUriInput =
    referenceImageInputFormats.includes("data_uri");
  const supportsAnyVideoImageInput =
    supportsFirstFrame || supportsLastFrame || supportsVideoReferenceImages;

  const updateReferenceImageInputFormat = (
    format: "url" | "data_uri",
    enabled: boolean,
  ) => {
    const nextFormats = new Set(
      referenceImageInputFormats.filter(
        (value) => value === "url" || value === "data_uri",
      ),
    );
    if (enabled) {
      nextFormats.add(format);
    } else {
      nextFormats.delete(format);
    }
    emitChange({
      ...configObj,
      referenceImageInputFormats: Array.from(nextFormats),
      ...(format === "data_uri" ? { supportDataUriInput: enabled } : {}),
    });
  };

  if (fields.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-border/30 p-4 text-center text-xs text-muted-foreground">
        当前模型类型暂无可视化参数项，如需补充专有参数请切换到「原始 JSON」标签页。
      </p>
    );
  }

  const simpleFields = fields.filter(
    (field) => field.type === "number" || field.type === "range",
  );

  return (
    <div className="flex flex-col gap-3">
      {simpleFields.length > 0 && (
        <div className="flex flex-col gap-3 rounded-lg border border-border/30 bg-muted/20 p-3">
          {simpleFields.map((field) => {
            const value = configObj[field.key];
            const numValue =
              typeof value === "number"
                ? value
                : typeof value === "string"
                  ? parseFloat(value)
                  : undefined;
            const hasValue = numValue !== undefined && !Number.isNaN(numValue);
            const fieldId = `${idPrefix}-${field.key}`;

            if (field.type === "range") {
              return (
                <div key={field.key} className="flex min-w-0 flex-col gap-2">
                  <div className="flex items-center justify-between gap-2">
                    <Label htmlFor={fieldId} className="text-xs text-muted-foreground">
                      {field.label}
                    </Label>
                    <span className="min-w-[3ch] text-right font-mono text-xs text-foreground tabular-nums">
                      {hasValue ? numValue : "—"}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      id={fieldId}
                      type="range"
                      min={field.min ?? 0}
                      max={field.max ?? 1}
                      step={field.step ?? 0.1}
                      value={hasValue ? numValue : field.defaultValue ?? field.min ?? 0}
                      onChange={(event) =>
                        updateSimpleField(field.key, event.target.value)
                      }
                      className="h-2 flex-1 cursor-pointer accent-primary"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-xs"
                      disabled={!hasValue}
                      onClick={() => updateSimpleField(field.key, "")}
                      aria-label={`清除 ${field.label}`}
                      title="恢复默认"
                    >
                      <RotateCcw />
                    </Button>
                  </div>
                  {field.hint && (
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      {field.hint}
                    </p>
                  )}
                </div>
              );
            }

            return (
              <div key={field.key} className="flex min-w-0 flex-col gap-2">
                <Label htmlFor={fieldId} className="text-xs text-muted-foreground">
                  {field.label}
                </Label>
                <Input
                  id={fieldId}
                  type="number"
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  placeholder={field.placeholder}
                  value={hasValue ? numValue : ""}
                  onChange={(event) => updateSimpleField(field.key, event.target.value)}
                  className="font-mono text-xs"
                />
                {field.hint && (
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {field.hint}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}

      {fields.some((field) => field.type === "supported-sizes") && (
        <div className="rounded-lg border border-border/30 bg-muted/20 p-3">
          <SupportedSizesEditor
            value={configObj.supportedSizes as SizesMap | undefined}
            onChange={(value) => updateComplexField("supportedSizes", value)}
          />
        </div>
      )}

      {fields
        .filter((field) => field.type === "aspect-ratios")
        .map((field) => (
          <div
            key={field.key}
            className="rounded-lg border border-border/30 bg-muted/20 p-3"
          >
            <AspectRatiosEditor
              label={field.label}
              hint={field.hint}
              presetOptions={field.presetOptions}
              value={configObj[field.key] as string[] | undefined}
              onChange={(value) => updateComplexField(field.key, value)}
            />
          </div>
        ))}

      {modelType === 2 && normalizedPlatform !== "comfyui" && (
        <div className="flex flex-col gap-3 rounded-lg border border-border/30 bg-muted/20 p-3">
          <div className="flex flex-col gap-1">
            <p className="text-xs font-medium text-foreground">参考图能力</p>
            <p className="text-xs leading-relaxed text-muted-foreground">
              控制生图时是否支持传入参考图以及允许的参考图张数。
            </p>
          </div>

          <ToggleSettingCard
            checked={supportsImageReferenceInputs}
            title="支持参考图输入"
            description="开启后可在生图时传入参考图；关闭时仅支持文生图。"
            onToggle={() => {
              const nextEnabled = !supportsImageReferenceInputs;
              emitChange({
                ...configObj,
                supportReferenceImages: nextEnabled,
                minReferenceImages: nextEnabled ? configObj.minReferenceImages : 0,
                maxReferenceImages: nextEnabled ? configObj.maxReferenceImages : 0,
                referenceImageInputFormats: nextEnabled
                  ? configObj.referenceImageInputFormats
                  : [],
                supportDataUriInput: nextEnabled ? configObj.supportDataUriInput : false,
              });
            }}
          />

          <div className="grid gap-2.5 sm:grid-cols-2">
            <ToggleSettingCard
              checked={supportsReferenceImageUrlInput}
              title="允许 URL 传递"
              description="已配置公网存储或可公网访问地址时，直接传递图片链接。"
              disabled={!supportsImageReferenceInputs}
              onToggle={() =>
                updateReferenceImageInputFormat("url", !supportsReferenceImageUrlInput)
              }
            />
            <ToggleSettingCard
              checked={supportsReferenceImageDataUriInput}
              title="允许 Base64 / Data URI"
              description="无公网访问地址时，由服务端读取图片并编码后提交。"
              disabled={!supportsImageReferenceInputs}
              onToggle={() =>
                updateReferenceImageInputFormat(
                  "data_uri",
                  !supportsReferenceImageDataUriInput,
                )
              }
            />
          </div>

          <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
            <CapabilityNumberField
              label="最少参考图数量"
              value={getConfigNumberValue(configObj.minReferenceImages)}
              onChange={(value) => updateSimpleField("minReferenceImages", value)}
              min={0}
              step={1}
              disabled={!supportsImageReferenceInputs}
              placeholder="例如：0"
              hint="通常填 0；仅在模型强制要求至少上传多张参考图时填写。"
            />
            <CapabilityNumberField
              label="最多参考图数量"
              value={getConfigNumberValue(configObj.maxReferenceImages)}
              onChange={(value) => updateSimpleField("maxReferenceImages", value)}
              min={0}
              step={1}
              disabled={!supportsImageReferenceInputs}
              placeholder="例如：3"
              hint="限制单次生图最多可传入的参考图张数。"
            />
          </div>
        </div>
      )}

      {modelType === 2 && normalizedPlatform === "openai_compatible" && (
        <div className="flex flex-col gap-3 rounded-lg border border-border/30 bg-muted/20 p-3">
          <div className="flex flex-col gap-1">
            <p className="text-xs font-medium text-foreground">OpenAI 兼容图片任务模式</p>
            <p className="text-xs leading-relaxed text-muted-foreground">
              开启后按异步任务提交并轮询生成结果；关闭时保持同步生成接口调用。
            </p>
          </div>

          <ToggleSettingCard
            checked={supportsAsyncImageTaskMode}
            title="启用异步任务模式"
            description="适用于先返回任务编号、再通过查询接口获取图片的兼容服务。"
            onToggle={() => updateComplexField("asyncMode", !supportsAsyncImageTaskMode)}
          />
        </div>
      )}

      {modelType === 3 && normalizedPlatform !== "comfyui" && (
        <div className="flex flex-col gap-3 rounded-lg border border-border/30 bg-muted/20 p-3">
          <div className="flex flex-col gap-1">
            <p className="text-xs font-medium text-foreground">多模态输入能力</p>
            <p className="text-xs leading-relaxed text-muted-foreground">
              控制视频生成支持的首尾帧、参考图、参考视频、参考音频及其数量上限。
            </p>
          </div>

          <div className="grid gap-2.5 [grid-template-columns:repeat(auto-fit,minmax(240px,1fr))]">
            <ToggleSettingCard
              checked={supportsFirstFrame}
              title="支持首帧图"
              description="支持上传首帧图以锁定开场画面。"
              onToggle={() => updateComplexField("supportFirstFrame", !supportsFirstFrame)}
            />
            <ToggleSettingCard
              checked={supportsLastFrame}
              title="支持尾帧图"
              description="支持上传尾帧图以约束结尾画面。"
              onToggle={() => updateComplexField("supportLastFrame", !supportsLastFrame)}
            />
            <ToggleSettingCard
              checked={supportsVideoReferenceImages}
              title="支持参考图"
              description="支持上传多张参考图，用于固定角色、场景或画面风格。"
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
              description="支持上传参考视频，用于动作或镜头风格参考。"
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
              description="支持上传参考音频，用于节奏或音频条件参考。"
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

          <div className="flex flex-col gap-2">
            <div className="flex flex-col gap-1">
              <p className="text-xs font-medium text-foreground">图片传递方式</p>
              <p className="text-xs leading-relaxed text-muted-foreground">
                同时作用于首帧图、尾帧图与参考图；至少启用一种方式才能提交图片输入。
              </p>
            </div>
            <div className="grid gap-2.5 sm:grid-cols-2">
              <ToggleSettingCard
                checked={supportsReferenceImageUrlInput}
                title="允许 URL 传递"
                description="有公网存储或可公网访问地址时，直接传递图片链接。"
                disabled={!supportsAnyVideoImageInput}
                onToggle={() =>
                  updateReferenceImageInputFormat("url", !supportsReferenceImageUrlInput)
                }
              />
              <ToggleSettingCard
                checked={supportsReferenceImageDataUriInput}
                title="允许 Base64 / Data URI"
                description="无公网访问地址时，由服务端读取图片并编码后提交。"
                disabled={!supportsAnyVideoImageInput}
                onToggle={() =>
                  updateReferenceImageInputFormat(
                    "data_uri",
                    !supportsReferenceImageDataUriInput,
                  )
                }
              />
            </div>
          </div>

          <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
            <CapabilityNumberField
              label="最少图片输入数"
              value={getConfigNumberValue(configObj.minImageInputs)}
              onChange={(value) => updateSimpleField("minImageInputs", value)}
              min={0}
              step={1}
              placeholder="例如：1"
              hint="统计范围包含首帧图、尾帧图与全部参考图。"
            />
            <CapabilityNumberField
              label="最多图片输入数"
              value={getConfigNumberValue(configObj.maxImageInputs)}
              onChange={(value) => updateSimpleField("maxImageInputs", value)}
              min={0}
              step={1}
              placeholder="例如：3"
              hint="限制单次图片类输入总数，避免首尾帧与参考图合计超限。"
            />
            <CapabilityNumberField
              label="最多参考图数量"
              value={getConfigNumberValue(configObj.maxReferenceImages)}
              onChange={(value) => updateSimpleField("maxReferenceImages", value)}
              min={0}
              step={1}
              disabled={!supportsVideoReferenceImages}
              placeholder="例如：3"
              hint="单次可传入参考图张数上限。"
            />
            <CapabilityNumberField
              label="最多参考视频数量"
              value={getConfigNumberValue(configObj.maxReferenceVideos)}
              onChange={(value) => updateSimpleField("maxReferenceVideos", value)}
              min={0}
              step={1}
              disabled={!supportsReferenceVideos}
              placeholder="例如：1"
              hint="单次可传入参考视频个数上限。"
            />
            <CapabilityNumberField
              label="最多参考音频数量"
              value={getConfigNumberValue(configObj.maxReferenceAudios)}
              onChange={(value) => updateSimpleField("maxReferenceAudios", value)}
              min={0}
              step={1}
              disabled={!supportsReferenceAudios}
              placeholder="例如：1"
              hint="单次可传入参考音频个数上限。"
            />
          </div>
        </div>
      )}

      {showReasoningConfig && isOpenAiReasoningPlatform(normalizedPlatform) && (
        <div className="flex flex-col gap-3 rounded-lg border border-border/30 bg-muted/20 p-3">
          <div className="flex flex-col gap-1">
            <p className="text-xs font-medium text-foreground">思考内容返回</p>
            <p className="text-xs leading-relaxed text-muted-foreground">
              自动模式下，开启“支持思考”或填写思考参数时会自动请求思考过程。
            </p>
          </div>

          <div className="flex min-w-0 flex-col gap-2">
            <Label className="text-xs text-muted-foreground">返回策略</Label>
            <Select
              value={includeReasoningMode}
              onValueChange={(value) => updateSelectField("includeReasoning", String(value))}
              items={[
                { value: "auto", label: "自动" },
                { value: "true", label: "显式返回思考内容" },
                { value: "false", label: "显式关闭思考内容返回" },
              ]}
            >
              <SelectTrigger aria-label="思考内容返回策略" className="w-full text-sm">
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

          <CapabilityNumberField
            label="思考预算（Thinking Budget）"
            value={getPositiveNumberValue(
              typeof configObj.thinkingBudget === "number"
                ? configObj.thinkingBudget
                : undefined,
            )}
            onChange={(value) => updateSimpleField("thinkingBudget", value)}
            min={0}
            step={1}
            placeholder="例如：2048"
            hint="适用于支持按 token 预算控制思考深度的 OpenAI 兼容服务。"
          />
        </div>
      )}

      {showReasoningConfig && isAnthropicReasoningPlatform(normalizedPlatform) && (
        <div className="rounded-lg border border-border/30 bg-muted/20 p-3">
          <CapabilityNumberField
            label="思考预算（Thinking Budget）"
            value={getPositiveNumberValue(
              typeof configObj.thinkingBudget === "number"
                ? configObj.thinkingBudget
                : undefined,
            )}
            onChange={(value) => updateSimpleField("thinkingBudget", value)}
            min={0}
            step={1}
            placeholder="例如：1024"
            hint="开启“支持思考”后，留空时默认使用 1024；如需自定义结构可在「原始 JSON」标签页中覆盖。"
          />
        </div>
      )}

      {showReasoningConfig && isDashScopeReasoningPlatform(normalizedPlatform) && (
        <div className="rounded-lg border border-border/30 bg-muted/20 p-3">
          <p className="text-xs leading-relaxed text-muted-foreground">
            DashScope 的思考模式由「能力与限制」标签页中的“支持思考”开关控制；如需补充平台专有参数，可切换到「原始 JSON」标签页设置。
          </p>
        </div>
      )}
    </div>
  );
}
