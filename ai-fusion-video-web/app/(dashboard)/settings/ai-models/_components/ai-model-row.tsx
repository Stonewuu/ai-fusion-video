"use client";

import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Edit2,
  FileText,
  Headphones,
  ImageIcon,
  Loader2,
  Power,
  Sparkles,
  Star,
  Trash2,
  Video,
  Volume2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getModelDisplayParts } from "@/lib/model-display";
import {
  generationCapabilityError,
  resolvedGenerationConfig,
} from "@/lib/generation-capabilities";
import type {
  AiModel,
  ApiConfig,
  ModelPreset,
  MultimodalInputType,
} from "@/lib/api/ai-model";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ModelVendorIcon } from "@/components/dashboard/model-vendor-icon";
import {
  buildGenerationCapabilityView,
  findCapabilityPreset,
  formatSemanticLabel,
  getApiConfigDefaultProtocol,
  getCapabilityChipClassName,
  MODEL_PROTOCOL_LABELS,
} from "./model-config-support";

export interface ModelConnectivityTestState {
  ok: boolean;
  durationMs?: number;
  summary: string;
}

const MULTIMODAL_META: Record<
  MultimodalInputType,
  { label: string; icon: typeof ImageIcon }
> = {
  image: { label: "图片", icon: ImageIcon },
  video: { label: "视频", icon: Video },
  audio: { label: "音频", icon: Headphones },
  file: { label: "文件", icon: FileText },
};

function formatContextWindow(tokens: number): string {
  if (tokens >= 1_000_000) {
    const millions = tokens / 1_000_000;
    return `${Number.isInteger(millions) ? millions : millions.toFixed(1)}M`;
  }
  if (tokens >= 1_000) {
    const thousands = tokens / 1_000;
    return `${Number.isInteger(thousands) ? thousands : thousands.toFixed(1)}K`;
  }
  return String(tokens);
}

interface AiModelRowProps {
  model: AiModel;
  config: ApiConfig;
  modelPresets: ModelPreset[];
  testing: boolean;
  testResult?: ModelConnectivityTestState;
  settingDefault: boolean;
  togglingStatus: boolean;
  onTestModel: (model: AiModel) => void;
  onSetDefault: (model: AiModel) => void;
  onToggleStatus: (model: AiModel) => void;
  onEditModel: (model: AiModel) => void;
  onDeleteModel: (model: AiModel) => void;
}

export function AiModelRow({
  model,
  config,
  modelPresets,
  testing,
  testResult,
  settingDefault,
  togglingStatus,
  onTestModel,
  onSetDefault,
  onToggleStatus,
  onEditModel,
  onDeleteModel,
}: AiModelRowProps) {
  const modelDisplay = getModelDisplayParts(model);
  const matchedPreset = findCapabilityPreset(model, modelPresets);
  const effectiveModelConfig = resolvedGenerationConfig(model, modelPresets);
  const capabilityError = generationCapabilityError(model);
  const providerDefaultProtocol = getApiConfigDefaultProtocol(config, model.modelType);
  const effectiveModelProtocol = model.modelProtocol || providerDefaultProtocol;
  const isProtocolOverridden = Boolean(
    model.modelProtocol && model.modelProtocol !== providerDefaultProtocol,
  );
  const isProtocolMissing =
    model.modelType >= 1
    && model.modelType <= 3
    && config.platform !== "comfyui"
    && config.platform !== "sonilo"
    && !effectiveModelProtocol;
  const capabilityView = capabilityError
    ? null
    : buildGenerationCapabilityView(model, effectiveModelConfig);

  const canTest =
    model.modelType === 1
    || (model.modelType === 4
      && ["gemini", "vertex_ai", "vertex_ai_api_key"].includes(config.platform || ""));
  const isDisabled = model.status === 0;
  const canSetDefault = !model.defaultModel && !isDisabled && config.status === 1;

  const workflowLabel = model.comfyuiWorkflowId
    ? effectiveModelConfig.comfyuiWorkflowVersionNo
      ? `工作流 v${effectiveModelConfig.comfyuiWorkflowVersionNo}`
      : "已发布工作流"
    : null;
  const presetLabel = !workflowLabel
    ? matchedPreset?.name || model.capabilityPresetCode || null
    : null;

  const multimodalTitle = model.multimodalInputTypes
    .map((inputType) => {
      const label = MULTIMODAL_META[inputType]?.label ?? inputType;
      const transports = (model.multimodalInputTransports[inputType] ?? [])
        .map((item) => (item === "url" ? "URL" : "Base64"))
        .join("/");
      return transports ? `${label}（${transports}）` : label;
    })
    .join("、");

  return (
    <div
      className={cn(
        "group/model flex flex-col gap-2 rounded-lg px-3 py-2.5 transition-colors hover:bg-muted/35 sm:flex-row sm:items-start sm:gap-3",
        isDisabled && "opacity-65",
      )}
    >
      <div className="flex min-w-0 flex-1 items-start gap-3">
        <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-muted/60">
          <ModelVendorIcon source={model} className="size-4" />
        </div>

        <div className="min-w-0 flex-1 space-y-1.5">
          {/* 主行：名称、模型 ID 与状态 */}
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <p className="text-sm font-medium text-foreground break-words">
              {modelDisplay.name}
            </p>
            {modelDisplay.code && (
              <span className="font-mono text-xs text-muted-foreground break-all">
                {modelDisplay.code}
              </span>
            )}
            {model.defaultModel && (
              <Badge
                variant="secondary"
                className="bg-amber-500/12 text-amber-700 dark:text-amber-400"
              >
                <Star className="size-3" />
                默认
              </Badge>
            )}
            {isDisabled && (
              <Badge variant="outline" className="text-muted-foreground">
                已停用
              </Badge>
            )}
            {testResult && (
              <span
                title={testResult.summary}
                className={cn(
                  "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium tabular-nums",
                  testResult.ok
                    ? "bg-emerald-500/12 text-emerald-700 dark:text-emerald-400"
                    : "bg-destructive/10 text-destructive",
                )}
              >
                {testResult.ok ? (
                  <CheckCircle2 className="size-3" />
                ) : (
                  <AlertCircle className="size-3" />
                )}
                {testResult.ok
                  ? typeof testResult.durationMs === "number"
                    ? `${testResult.durationMs} ms`
                    : "连通正常"
                  : "检测失败"}
              </span>
            )}
          </div>

          {/* 副行：精简元信息与关键能力 */}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            {workflowLabel && <span>{workflowLabel}</span>}
            {presetLabel && (
              <span title={`能力预设：${model.capabilityPresetCode}`}>
                预设 · {presetLabel}
              </span>
            )}
            {isProtocolMissing ? (
              <span className="inline-flex items-center gap-1 font-medium text-destructive">
                <AlertCircle className="size-3.5" />
                请求协议未配置
              </span>
            ) : isProtocolOverridden ? (
              <span
                className="text-violet-700 dark:text-violet-400"
                title="模型已覆盖所属 API 配置的默认协议"
              >
                协议 · {formatSemanticLabel(effectiveModelProtocol, MODEL_PROTOCOL_LABELS)}
              </span>
            ) : null}

            {model.contextWindow && model.contextWindow > 0 ? (
              <span
                className="tabular-nums"
                title={`上下文窗口：${model.contextWindow.toLocaleString()} tokens`}
              >
                {formatContextWindow(model.contextWindow)} 上下文
              </span>
            ) : null}

            {model.supportReasoning && (
              <span
                className="inline-flex items-center gap-1 text-sky-700 dark:text-sky-400"
                title={
                  model.reasoningEffortLevels.length > 0
                    ? `支持思考，可选等级：${model.reasoningEffortLevels.join(" / ")}`
                    : "支持思考推理"
                }
              >
                <Sparkles className="size-3" />
                思考
                {model.reasoningEffortLevels.length > 0
                  ? `（${model.reasoningEffortLevels.length} 档）`
                  : ""}
              </span>
            )}

            {model.multimodalInputTypes.length > 0 && (
              <span
                className="inline-flex items-center gap-1.5 text-emerald-700 dark:text-emerald-400"
                title={`多模态输入：${multimodalTitle}`}
              >
                {model.multimodalInputTypes.map((inputType) => {
                  const Icon = MULTIMODAL_META[inputType]?.icon ?? ImageIcon;
                  return <Icon key={inputType} className="size-3.5" aria-hidden="true" />;
                })}
                <span>多模态</span>
              </span>
            )}
          </div>

          {capabilityView && capabilityView.chips.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {capabilityView.chips.map((chip) => (
                <span
                  key={`${model.id}-${chip.label}`}
                  className={cn(
                    "inline-flex items-center rounded-full border px-2 py-0.5 text-xs",
                    getCapabilityChipClassName(chip.tone),
                  )}
                >
                  {chip.label}
                </span>
              ))}
            </div>
          )}

          {capabilityError && (
            <p className="flex items-start gap-1.5 text-xs text-destructive">
              <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
              <span>{capabilityError}</span>
            </p>
          )}
        </div>
      </div>

      {/* 右侧操作区：统一 32px（sm / icon-sm）高度 */}
      <div className="flex flex-wrap items-center gap-1.5 sm:shrink-0">
        {canTest && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => onTestModel(model)}
            disabled={testing || isDisabled || config.status === 0}
            title={
              isDisabled || config.status === 0
                ? "请先启用服务与模型后再检测"
                : model.modelType === 4
                  ? "打开音频试听"
                  : "发送连通性测试"
            }
          >
            {testing ? (
              <Loader2 className="animate-spin" />
            ) : model.modelType === 4 ? (
              <Volume2 />
            ) : (
              <Activity />
            )}
            {model.modelType === 4 ? "试听" : testing ? "检测中" : "检测"}
          </Button>
        )}

        {canSetDefault && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={settingDefault}
            onClick={() => onSetDefault(model)}
          >
            {settingDefault ? <Loader2 className="animate-spin" /> : <Star />}
            设为默认
          </Button>
        )}

        <div className="flex items-center gap-1 text-muted-foreground">
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            disabled={togglingStatus}
            onClick={() => onToggleStatus(model)}
            aria-label={isDisabled ? `启用模型 ${modelDisplay.name}` : `停用模型 ${modelDisplay.name}`}
            title={isDisabled ? "启用模型" : "停用模型"}
          >
            {togglingStatus ? <Loader2 className="animate-spin" /> : <Power />}
          </Button>

          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            onClick={() => onEditModel(model)}
            aria-label={`编辑模型 ${modelDisplay.name}`}
            title="编辑模型"
          >
            <Edit2 />
          </Button>

          <Button
            type="button"
            variant="destructive-ghost"
            size="icon-sm"
            onClick={() => onDeleteModel(model)}
            aria-label={`删除模型 ${modelDisplay.name}`}
            title="删除模型"
          >
            <Trash2 />
          </Button>
        </div>
      </div>
    </div>
  );
}
