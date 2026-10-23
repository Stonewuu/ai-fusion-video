"use client";

import { ChevronRight, CloudDownload, Edit2, Loader2, Plus, Search, Star, Trash2, Workflow } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/utils";
import { toastApiError } from "@/lib/api/toast-api-error";
import { getModelDisplayParts } from "@/lib/model-display";
import { generationCapabilityError, resolvedGenerationConfig } from "@/lib/generation-capabilities";
import { aiModelApi, PLATFORM_LABELS, type AiModel, type ApiConfig, type ModelPreset } from "@/lib/api/ai-model";
import { Button } from "@/components/ui/button";
import { ModelVendorIcon, ProviderVendorIcon } from "@/components/dashboard/model-vendor-icon";
import { maskSecret } from "../../_shared";
import type { ProviderDefinition } from "@/lib/api/provider-catalog";
import { SponsorNotice } from "./sponsor-notice";
import { buildGenerationCapabilityView, findCapabilityPreset, formatSemanticLabel, getApiConfigDefaultProtocol, getCapabilityChipClassName, MODEL_PROTOCOL_LABELS } from "./model-config-support";

interface ApiConfigCardProps {
  config: ApiConfig;
  provider?: ProviderDefinition;
  models: AiModel[];
  modelPresets: ModelPreset[];
  collapsed: boolean;
  testingModelIds: Set<number>;
  onToggle: () => void;
  onFetch: () => void;
  onWorkflow: () => void;
  onEditConfig: () => void;
  onDeleteConfig: () => void;
  onEditModel: (model: AiModel) => void;
  onDeleteModel: (id: number) => void;
  onAddModel: () => void;
  onTestTextModel: (model: AiModel) => void;
  onModelsChanged: () => Promise<void>;
}

export function ApiConfigCard({ config, provider, models, modelPresets, collapsed, testingModelIds, onToggle, onFetch, onWorkflow, onEditConfig, onDeleteConfig, onEditModel, onDeleteModel, onAddModel, onTestTextModel, onModelsChanged }: ApiConfigCardProps) {
  const shouldReduceMotion = useReducedMotion();
    const configModels = models.filter(m => m.apiConfigId === config.id);

    // 按类型分组
    const modelTypeGroups = [
      { type: 2, label: "图像生成" },
      { type: 3, label: "视频生成" },
      { type: 1, label: "对话" },
    ].map(g => ({
      ...g,
      models: configModels.filter(m => m.modelType === g.type),
    })).filter(g => g.models.length > 0);

    return (
      <div
        key={config.id}
        data-api-config-card
        className={cn(
          "rounded-xl border overflow-hidden",
          "bg-card/50 backdrop-blur-sm border-border/50"
        )}
      >
        {/* ── API 配置头部 ── */}
        <div className={cn(
          "flex items-start gap-3 px-4 py-3 group bg-muted/30"
        )}>
          <button
            type="button"
            onClick={() => onToggle()}
            aria-expanded={!collapsed}
            title={collapsed ? "展开 API 配置" : "折叠 API 配置"}
            className={cn(
              "mt-1 h-7 w-7 rounded-md flex items-center justify-center shrink-0",
              "text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
            )}
          >
            <ChevronRight className={cn(
              "h-3.5 w-3.5 transition-transform duration-200 motion-reduce:transition-none",
              !collapsed && "rotate-90"
            )} />
          </button>
          <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-muted/60">
            <ProviderVendorIcon
              provider={provider?.vendor ?? config.providerId ?? config.platform}
              className="size-5"
            />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <p className="text-sm font-medium break-words">{config.name}</p>
              {provider?.sponsored && <span className="rounded-full bg-primary/10 px-2 py-1 text-xs text-primary">赞助合作伙伴</span>}
              {config.platform && (
                <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px] text-muted-foreground">
                  {PLATFORM_LABELS[config.platform] || config.platform}
                </span>
              )}
              <div className={cn(
                "w-1.5 h-1.5 rounded-full shrink-0",
                config.status === 1 ? "bg-green-400" : "bg-muted-foreground/30"
              )} />
              {config.status === 0 && <span className="text-xs text-muted-foreground">已停用</span>}
            </div>
            {provider?.sponsored && <SponsorNotice provider={provider} />}
            <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5 flex-wrap">
              {config.apiKey && (
                <span className="font-mono text-[10px]">{maskSecret(config.apiKey)}</span>
              )}
              {config.proxyType && config.proxyType !== "none" && config.proxyHost && config.proxyPort && (
                <span className="font-mono text-[10px] px-1 py-0.5 rounded bg-sky-500/10 text-sky-500 break-all">
                  {config.proxyType}://{config.proxyHost}:{config.proxyPort}{config.proxyUsername ? " auth" : ""}
                </span>
              )}
              <span>{configModels.length} 个模型</span>
              {([
                { label: "文", protocol: config.textProtocol },
                { label: "图", protocol: config.imageProtocol },
                { label: "视", protocol: config.videoProtocol },
              ] as const).map(item => item.protocol ? (
                <span
                  key={item.label}
                  className="rounded bg-violet-500/10 px-1 py-0.5 text-[10px] text-violet-600"
                  title={`${item.label}类默认：${formatSemanticLabel(item.protocol, MODEL_PROTOCOL_LABELS)}`}
                >
                  {item.label} · {formatSemanticLabel(item.protocol, MODEL_PROTOCOL_LABELS)}
                </span>
              ) : null)}
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0 mt-0.5">
            {config.platform !== "comfyui" && config.platform !== "sonilo" && (
              <Button variant="ghost" size="icon-sm"
                type="button"
                onClick={() => { onFetch() }}
                title="获取可用模型列表"
              >
                <CloudDownload className="h-3.5 w-3.5" />
              </Button>
            )}
            {config.platform === "comfyui" && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                                onClick={() => onWorkflow()}
                title="管理 ComfyUI 工作流"
                aria-label={`管理 ${config.name} 的 ComfyUI 工作流`}
              >
                <span className="inline-flex items-center gap-1.5 text-primary">
                  <Workflow />
                  管理工作流
                </span>
              </Button>
            )}
            <Button variant="ghost" size="icon-sm"
              type="button"
              onClick={() => { onEditConfig() }}
              title="编辑 API 配置"
            >
              <Edit2 className="h-3.5 w-3.5" />
            </Button>
            <Button variant="destructive-ghost" size="icon-sm"
              type="button"
              onClick={() => onDeleteConfig()}
              title="删除 API 配置"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        <AnimatePresence initial={false}>
          {!collapsed && (
            <motion.div
              key="config-models"
              initial={shouldReduceMotion ? false : { height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={shouldReduceMotion
                ? { duration: 0 }
                : { duration: 0.24, ease: [0.25, 0.46, 0.45, 0.94] }}
              className="overflow-hidden border-t border-border/40"
            >
              <div className="px-4 py-2 pr-3">
            {modelTypeGroups.length === 0 ? (
              <p className="text-xs text-muted-foreground/50 text-center py-3">
                暂无模型
              </p>
            ) : (
              modelTypeGroups.map((group, gi) => {
                const defaultModel = group.models.find(m => m.defaultModel);

                return (
                  <div key={group.type} className={cn(gi > 0 && "mt-2 pt-2 border-t border-border/30")}>
                    {/* 类型分组标题 */}
                    <div className="flex items-center gap-2 px-1 py-1">
                      <span className="text-[11px] font-medium text-muted-foreground shrink-0">
                        {group.label}
                      </span>
                      <div className="flex-1 h-px bg-border/15" />
                      {defaultModel ? (
                        <span
                          className="inline-flex min-w-0 max-w-[55%] items-center gap-1 text-[10px] text-amber-500/80 font-medium shrink-0"
                          title={`默认: ${defaultModel.name}`}
                        >
                          <Star className="h-2.5 w-2.5" />
                          <span className="truncate">默认: {defaultModel.name}</span>
                        </span>
                      ) : (
                        <span className="text-[10px] text-muted-foreground/30 shrink-0">未设默认</span>
                      )}
                    </div>

                    {/* 模型行 */}
                    {group.models.map((model) => {
                      const modelDisplay = getModelDisplayParts(model);
                      const matchedPreset = findCapabilityPreset(model, modelPresets);
                      const effectiveModelConfig = resolvedGenerationConfig(model, modelPresets);
                      const capabilityError = generationCapabilityError(model);
                      const capabilityPresetLabel = model.comfyuiWorkflowId
                        ? effectiveModelConfig.comfyuiWorkflowVersionNo ? `工作流 v${effectiveModelConfig.comfyuiWorkflowVersionNo}` : "已发布工作流"
                        : matchedPreset?.name
                        || model.capabilityPresetCode
                        || "自定义";
                      const providerDefaultProtocol = getApiConfigDefaultProtocol(config, model.modelType);
                      const effectiveModelProtocol = model.modelProtocol || providerDefaultProtocol;
                      const protocolSource = model.modelProtocol
                        ? "模型覆盖"
                        : providerDefaultProtocol
                          ? "API 配置默认"
                          : "未配置";
                      const capabilityView = capabilityError ? { chips: [], summary: capabilityError }
                        : buildGenerationCapabilityView(model, effectiveModelConfig);

                      return (
                        <div
                          key={model.id}
                          className="flex items-start gap-3 px-3 py-2.5 rounded-lg group/model hover:bg-muted/30 transition-colors"
                        >
                          <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-muted/60">
                            <ModelVendorIcon source={model} className="size-4" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <p className="text-sm font-medium break-words">{modelDisplay.name}</p>
                              {model.defaultModel && (
                                <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-amber-500/10 text-[10px] text-amber-500 font-medium">
                                  <Star className="h-2.5 w-2.5" />
                                  默认
                                </span>
                              )}
                              <div className={cn(
                                "w-1.5 h-1.5 rounded-full shrink-0",
                                model.status === 1 ? "bg-green-400" : "bg-muted-foreground/30"
                              )} />
                            </div>
                            <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground mt-0.5 flex-wrap">
                              {modelDisplay.code && (
                                <span className="font-mono px-1 py-0.5 rounded bg-muted/40 break-all">
                                  {modelDisplay.code}
                                </span>
                              )}
                              <span
                                className={cn(
                                  "px-1 py-0.5 rounded",
                                  model.capabilityPresetCode
                                    ? "bg-amber-500/10 text-amber-600"
                                    : "bg-muted/50 text-muted-foreground"
                                )}
                                title={model.capabilityPresetCode
                                  ? `模型能力预设：${model.capabilityPresetCode}`
                                  : "模型能力由自定义高级设置提供"}
                              >
                                能力：{capabilityPresetLabel}
                              </span>
                              <span
                                className={cn(
                                  "px-1 py-0.5 rounded",
                                  effectiveModelProtocol
                                    ? "bg-violet-500/10 text-violet-600"
                                    : "bg-destructive/10 text-destructive"
                                )}
                                title={protocolSource}
                              >
                                {effectiveModelProtocol
                                  ? `${formatSemanticLabel(effectiveModelProtocol, MODEL_PROTOCOL_LABELS)} · ${protocolSource}`
                                  : "请求协议未配置"}
                              </span>
                              {model.supportReasoning && (
                                <span className="px-1 py-0.5 rounded bg-sky-500/10 text-sky-500">思考</span>
                              )}
                              {model.multimodalInputTypes.map((inputType) => {
                                const transportLabel = (model.multimodalInputTransports[inputType] ?? [])
                                  .map((item) => item === "url" ? "URL" : "Base64")
                                  .join("/");
                                return (
                                  <span
                                    key={`${model.id}-${inputType}`}
                                    className="px-1 py-0.5 rounded bg-emerald-500/10 text-emerald-500"
                                    title={`${transportLabel} 输入`}
                                  >
                                    {{ image: "图片", video: "视频", audio: "音频", file: "文件" }[inputType]}
                                    {transportLabel ? ` · ${transportLabel}` : ""}
                                  </span>
                                );
                              })}
                              {model.contextWindow && model.contextWindow > 0 && (
                                <span className="px-1 py-0.5 rounded bg-muted/50 text-muted-foreground">
                                  上下文：{model.contextWindow.toLocaleString()}
                                </span>
                              )}
                            </div>
                            {capabilityView && (
                              <div className="mt-1.5 space-y-1.5">
                                {capabilityView.chips.length > 0 && (
                                  <div className="flex flex-wrap gap-1">
                                    {capabilityView.chips.map(chip => (
                                      <span
                                        key={`${model.id}-${chip.label}`}
                                        className={cn(
                                          "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] leading-none",
                                          getCapabilityChipClassName(chip.tone)
                                        )}
                                      >
                                        {chip.label}
                                      </span>
                                    ))}
                                  </div>
                                )}
                                <p className="text-[10px] text-muted-foreground/75 leading-4">
                                  {capabilityView.summary}
                                </p>
                              </div>
                            )}
                          </div>

                          {/* 操作按钮 */}
                          <div className="flex items-center gap-2 shrink-0 mt-0.5">
                            {model.modelType === 1 && (
                              <Button variant="outline" size="xs"
                                onClick={() => onTestTextModel(model)}
                                disabled={testingModelIds.has(model.id)}
                              >
                                {testingModelIds.has(model.id) ? (
                                  <Loader2 className="h-3 w-3 animate-spin" />
                                ) : (
                                  <Search className="h-3 w-3" />
                                )}
                                {testingModelIds.has(model.id) ? "检测中" : "检测"}
                              </Button>
                            )}
                            {!model.defaultModel && (
                              <Button variant="outline" size="xs"
                                onClick={async () => {
                                  try {
                                    const sameTypeModels = models.filter(m => m.modelType === model.modelType && m.defaultModel);
                                    for (const dm of sameTypeModels) {
                                      await aiModelApi.update({ id: dm.id, defaultModel: false });
                                    }
                                    await aiModelApi.update({ id: model.id, defaultModel: true });
                                    await onModelsChanged();
                                  } catch (err) {
                                    console.error("设置默认模型失败:", err);
                                    toastApiError(err, "设置默认模型失败");
                                  }
                                }}
                              >
                                <Star className="h-3 w-3" />
                                设为默认
                              </Button>
                            )}
                            <Button variant="ghost" size="icon-sm"
                              onClick={() => { onEditModel(model) }}
                              title="编辑模型"
                            >
                              <Edit2 className="h-3 w-3" />
                            </Button>
                            <Button variant="destructive-ghost" size="icon-sm"
                              onClick={() => onDeleteModel(model.id)}
                              title="删除模型"
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                );
              })
            )}

            {/* 添加模型按钮 */}
            <Button variant="outline" size="sm" className="w-full justify-start"
              onClick={() => {
                onAddModel();
              }}
            >
              <Plus className="h-3.5 w-3.5" />
              添加模型
            </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  }
