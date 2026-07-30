"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  ChevronRight,
  CloudDownload,
  Download,
  Edit2,
  LayoutGrid,
  List,
  Loader2,
  Plus,
  Search,
  Settings2,
  Star,
  Trash2,
  Upload,
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { toastApiError } from "@/lib/api/toast-api-error";
import { getModelDisplayParts } from "@/lib/model-display";
import {
  aiModelApi,
  apiConfigApi,
  PLATFORM_LABELS,
  type AiModel,
  type ApiConfig,
  type ModelPreset,
} from "@/lib/api/ai-model";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
  ModelVendorIcon,
  ProviderVendorIcon,
} from "@/components/dashboard/model-vendor-icon";
import {
  containerVariants,
  itemVariants,
  maskSecret,
} from "../_shared";
import { AiModelDialog } from "./_components/ai-model-dialog";
import { ApiConfigDialog } from "./_components/api-config-dialog";
import { FetchRemoteModelsDialog } from "./_components/fetch-remote-models-dialog";
import {
  buildGenerationCapabilityView,
  findCapabilityPreset,
  formatSemanticLabel,
  getApiConfigDefaultProtocol,
  getCapabilityChipClassName,
  mergeConfigObjects,
  MODEL_PROTOCOL_LABELS,
  parseConfigJson,
} from "./_components/model-config-support";

type AiModelViewMode = "list" | "card";

const AI_MODEL_VIEW_MODE_STORAGE_KEY = "fusion-ai-model-view-mode";

function isAiModelViewMode(value: string | null): value is AiModelViewMode {
  return value === "list" || value === "card";
}

export default function AiModelsPage() {
  // AI 模型列表
  const [models, setModels] = useState<AiModel[]>([]);
  const [modelPresets, setModelPresets] = useState<ModelPreset[]>([]);

  // API 配置列表
  const [configs, setConfigs] = useState<ApiConfig[]>([]);
  const [configsLoading, setConfigsLoading] = useState(true);

  // Dialog 状态
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState<ApiConfig | null>(null);
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [editingModel, setEditingModel] = useState<AiModel | null>(null);
  const [modelDialogApiConfigId, setModelDialogApiConfigId] = useState<number | undefined>(undefined);
  const [fetchModelsDialogOpen, setFetchModelsDialogOpen] = useState(false);
  const [fetchModelsConfig, setFetchModelsConfig] = useState<ApiConfig | null>(null);
  const [testingModelIds, setTestingModelIds] = useState<Set<number>>(new Set());
  const [viewMode, setViewMode] = useState<AiModelViewMode>("list");
  const [collapsedConfigIds, setCollapsedConfigIds] = useState<Set<number>>(new Set());
  const importInputRef = useRef<HTMLInputElement>(null);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportIncludeSecrets, setExportIncludeSecrets] = useState(false);

  const loadModels = useCallback(async () => {
    try {
      const data = await aiModelApi.list();
      setModels(data);
    } catch (err) {
      console.error("加载 AI 模型列表失败:", err);
      toastApiError(err, "加载 AI 模型列表失败");
    }
  }, []);

  const loadConfigs = useCallback(async () => {
    try {
      setConfigsLoading(true);
      const data = await apiConfigApi.list();
      setConfigs(data);
    } catch (err) {
      console.error("加载 API 配置列表失败:", err);
      toastApiError(err, "加载 API 配置列表失败");
    } finally {
      setConfigsLoading(false);
    }
  }, []);

  const loadModelPresets = useCallback(async () => {
    try {
      const data = await aiModelApi.presets();
      setModelPresets(data);
    } catch (err) {
      console.error("加载模型预设失败:", err);
      toastApiError(err, "加载模型预设失败");
    }
  }, []);

  useEffect(() => {
    loadModels();
    loadConfigs();
    loadModelPresets();
  }, [loadConfigs, loadModelPresets, loadModels]);

  useEffect(() => {
    const storedViewMode = window.localStorage.getItem(AI_MODEL_VIEW_MODE_STORAGE_KEY);
    if (isAiModelViewMode(storedViewMode)) {
      setViewMode(storedViewMode);
    }
  }, []);

  const handleTestTextModel = async (model: AiModel) => {
    setTestingModelIds((prev) => {
      const next = new Set(prev);
      next.add(model.id);
      return next;
    });

    try {
      const result = await aiModelApi.testTextConnectivity(model.id);
      const preview = result.responseText.replace(/\s+/g, " ").trim();
      const summary = preview.length > 120 ? `${preview.slice(0, 120)}...` : preview;
      const description = [
        summary ? `回复：${summary}` : "已收到模型响应",
        typeof result.durationMs === "number" ? `${result.durationMs} ms` : "",
      ].filter(Boolean).join(" · ");

      toast.success(`${result.modelName} 连通正常`, {
        description,
      });
    } catch (err) {
      toast.error(`${getModelDisplayParts(model).name} 检测失败`, {
        description: err instanceof Error ? err.message : "请求失败",
      });
    } finally {
      setTestingModelIds((prev) => {
        const next = new Set(prev);
        next.delete(model.id);
        return next;
      });
    }
  };

  const handleDeleteModel = async (id: number) => {
    if (!confirm("确定要删除该 AI 模型吗？")) return;
    try {
      await aiModelApi.delete(id);
      await loadModels();
    } catch (err) {
      console.error("删除模型失败:", err);
      toastApiError(err, "删除模型失败");
    }
  };

  const handleDeleteConfig = async (id: number) => {
    if (!confirm("确定要删除该 API 配置吗？")) return;
    try {
      await apiConfigApi.delete(id);
      await loadConfigs();
    } catch (err) {
      console.error("删除配置失败:", err);
      toastApiError(err, "删除配置失败");
    }
  };

  const handleViewModeChange = (mode: AiModelViewMode) => {
    setViewMode(mode);
    window.localStorage.setItem(AI_MODEL_VIEW_MODE_STORAGE_KEY, mode);
  };

  const handleToggleConfigCollapse = (configId: number) => {
    setCollapsedConfigIds(prev => {
      const next = new Set(prev);
      if (next.has(configId)) {
        next.delete(configId);
      } else {
        next.add(configId);
      }
      return next;
    });
  };

  const handleExport = (includeSecrets: boolean) => {
    const exportedConfigs = configs.map(config => ({
      ...config,
      ...(includeSecrets ? {} : { apiKey: undefined, appSecret: undefined, proxyPassword: undefined }),
    }));
    const payload = {
      version: 1,
      exportedAt: new Date().toISOString(),
      apiConfigs: exportedConfigs,
      models,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `ai-fusion-config-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    toast.success(includeSecrets ? "配置已导出（含密钥）" : "配置已导出");
    setExportDialogOpen(false);
  };

  const handleImport = async (file: File) => {
    try {
      const parsed = JSON.parse(await file.text()) as { apiConfigs?: ApiConfig[]; models?: AiModel[] };
      if (!Array.isArray(parsed.apiConfigs) || !Array.isArray(parsed.models)) throw new Error("配置文件结构不正确");
      const configIdMap = new Map<number, number>();
      for (const source of parsed.apiConfigs) {
        const createdId = await apiConfigApi.create({
          name: source.name,
          platform: source.platform || undefined,
          textProtocol: source.textProtocol || undefined,
          imageProtocol: source.imageProtocol || undefined,
          videoProtocol: source.videoProtocol || undefined,
          apiUrl: source.apiUrl || undefined,
          autoAppendV1Path: source.autoAppendV1Path,
          proxyType: source.proxyType || "none",
          proxyHost: source.proxyHost || undefined,
          proxyPort: source.proxyPort || undefined,
          proxyUsername: source.proxyUsername || undefined,
          proxyPassword: source.proxyPassword || undefined,
          apiKey: source.apiKey || undefined,
          appId: source.appId || undefined,
          appSecret: source.appSecret || undefined,
          status: source.status,
          remark: source.remark || undefined,
        });
        if (typeof source.id === "number") configIdMap.set(source.id, createdId);
      }
      for (const source of parsed.models) {
        const apiConfigId = source.apiConfigId ? configIdMap.get(source.apiConfigId) : undefined;
        if (!apiConfigId) continue;
        await aiModelApi.create({
          name: source.name,
          code: source.code,
          capabilityPresetCode: source.capabilityPresetCode || undefined,
          modelProtocol: source.modelProtocol || undefined,
          modelType: source.modelType,
          icon: source.icon || undefined,
          description: source.description || undefined,
          sort: source.sort,
          config: source.config || undefined,
          maxConcurrency: source.maxConcurrency || undefined,
          defaultModel: false,
          supportVision: source.supportVision,
          multimodalInputTypes: source.multimodalInputTypes,
          multimodalInputTransports: source.multimodalInputTransports,
          supportReasoning: source.supportReasoning,
          reasoningEffortLevels: source.reasoningEffortLevels ?? [],
          contextWindow: source.contextWindow || undefined,
          apiConfigId,
        });
      }
      await Promise.all([loadConfigs(), loadModels()]);
      toast.success("配置已导入", { description: "已创建新的 API 配置和模型，原配置不会被覆盖。" });
    } catch (error) {
      toast.error("导入失败", { description: error instanceof Error ? error.message : "请检查 JSON 文件" });
    } finally {
      if (importInputRef.current) importInputRef.current.value = "";
    }
  };

  const renderConfigCard = (config: ApiConfig) => {
    const configModels = models.filter(m => m.apiConfigId === config.id);
    const collapsed = collapsedConfigIds.has(config.id);

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
        className={cn(
          "rounded-xl border overflow-hidden",
          "bg-card/50 backdrop-blur-sm border-border/50",
          viewMode === "card" && "h-full"
        )}
      >
        {/* ── API 配置头部 ── */}
        <div className={cn(
          "flex items-start gap-3 px-4 py-3 group bg-muted/30",
          !collapsed && "border-b border-border/40"
        )}>
          <button
            type="button"
            onClick={() => handleToggleConfigCollapse(config.id)}
            aria-expanded={!collapsed}
            title={collapsed ? "展开 API 配置" : "折叠 API 配置"}
            className={cn(
              "mt-1 h-7 w-7 rounded-md flex items-center justify-center shrink-0",
              "text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
            )}
          >
            <ChevronRight className={cn(
              "h-3.5 w-3.5 transition-transform duration-200",
              !collapsed && "rotate-90"
            )} />
          </button>
          <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-muted/60">
            <ProviderVendorIcon
              provider={config.platform}
              className="size-5"
            />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <p className="text-sm font-medium break-words">{config.name}</p>
              {config.platform && (
                <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px] text-muted-foreground">
                  {PLATFORM_LABELS[config.platform] || config.platform}
                </span>
              )}
              <div className={cn(
                "w-1.5 h-1.5 rounded-full shrink-0",
                config.status === 1 ? "bg-green-400" : "bg-muted-foreground/30"
              )} />
            </div>
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
          <div className="flex items-center gap-1 shrink-0 mt-0.5">
            <button
              onClick={() => { setFetchModelsConfig(config); setFetchModelsDialogOpen(true); }}
              className="p-1.5 rounded-md text-sky-500 hover:text-sky-600 hover:bg-sky-500/10 transition-colors"
              title="获取可用模型列表"
            >
              <CloudDownload className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => { setEditingConfig(config); setConfigDialogOpen(true); }}
              className="p-1.5 rounded-md text-emerald-500 hover:text-emerald-600 hover:bg-emerald-500/10 transition-colors"
              title="编辑 API 配置"
            >
              <Edit2 className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => handleDeleteConfig(config.id)}
              className="p-1.5 rounded-md text-rose-500 hover:text-rose-600 hover:bg-rose-500/10 transition-colors"
              title="删除 API 配置"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {!collapsed && (
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
                      const effectiveModelConfig = mergeConfigObjects(
                        matchedPreset?.config || {},
                        parseConfigJson(model.config)
                      );
                      const capabilityPresetLabel = matchedPreset?.name
                        || model.capabilityPresetCode
                        || "自定义";
                      const providerDefaultProtocol = getApiConfigDefaultProtocol(config, model.modelType);
                      const effectiveModelProtocol = model.modelProtocol || providerDefaultProtocol;
                      const protocolSource = model.modelProtocol
                        ? "模型覆盖"
                        : providerDefaultProtocol
                          ? "API 配置默认"
                          : "未配置";
                      const capabilityView = buildGenerationCapabilityView(model, effectiveModelConfig);

                      return (
                        <div
                          key={model.id}
                          className="flex items-start gap-3 px-3 py-2.5 rounded-lg group/model hover:bg-white/5 transition-colors"
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
                                <p className="text-[10px] text-muted-foreground/75 leading-4">
                                  {capabilityView.summary}
                                </p>
                              </div>
                            )}
                          </div>

                          {/* 操作按钮 */}
                          <div className="flex items-center gap-1 shrink-0 mt-0.5">
                            {model.modelType === 1 && (
                              <button
                                onClick={() => handleTestTextModel(model)}
                                disabled={testingModelIds.has(model.id)}
                                className={cn(
                                  "flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium transition-all",
                                  "border border-sky-500/30 text-sky-500",
                                  "hover:bg-sky-500/10 hover:border-sky-500/50",
                                  "disabled:opacity-60 disabled:cursor-not-allowed"
                                )}
                              >
                                {testingModelIds.has(model.id) ? (
                                  <Loader2 className="h-3 w-3 animate-spin" />
                                ) : (
                                  <Search className="h-3 w-3" />
                                )}
                                {testingModelIds.has(model.id) ? "检测中" : "检测"}
                              </button>
                            )}
                            {!model.defaultModel && (
                              <button
                                onClick={async () => {
                                  try {
                                    const sameTypeModels = models.filter(m => m.modelType === model.modelType && m.defaultModel);
                                    for (const dm of sameTypeModels) {
                                      await aiModelApi.update({ id: dm.id, defaultModel: false });
                                    }
                                    await aiModelApi.update({ id: model.id, defaultModel: true });
                                    await loadModels();
                                  } catch (err) {
                                    console.error("设置默认模型失败:", err);
                                    toastApiError(err, "设置默认模型失败");
                                  }
                                }}
                                className={cn(
                                  "flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium transition-all",
                                  "border border-amber-500/30 text-amber-500",
                                  "hover:bg-amber-500/10 hover:border-amber-500/50"
                                )}
                              >
                                <Star className="h-3 w-3" />
                                设为默认
                              </button>
                            )}
                            <button
                              onClick={() => { setEditingModel(model); setModelDialogApiConfigId(undefined); setModelDialogOpen(true); }}
                              className="p-1 rounded-md text-emerald-500 hover:text-emerald-600 hover:bg-emerald-500/10 transition-colors"
                              title="编辑模型"
                            >
                              <Edit2 className="h-3 w-3" />
                            </button>
                            <button
                              onClick={() => handleDeleteModel(model.id)}
                              className="p-1 rounded-md text-rose-500 hover:text-rose-600 hover:bg-rose-500/10 transition-colors"
                              title="删除模型"
                            >
                              <Trash2 className="h-3 w-3" />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                );
              })
            )}

            {/* 添加模型按钮 */}
            <button
              onClick={() => {
                setEditingModel(null);
                setModelDialogApiConfigId(config.id);
                setModelDialogOpen(true);
              }}
              className={cn(
                "flex items-center gap-2 w-full px-3 py-2 mt-1 rounded-lg",
                "border border-dashed border-border/40 hover:border-primary/40",
                "text-xs text-muted-foreground/60 hover:text-primary",
                "transition-all duration-200"
              )}
            >
              <Plus className="h-3.5 w-3.5" />
              添加模型
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <motion.div
      className="w-full"
      variants={containerVariants}
      initial={false}
      animate="visible"
    >
      {/* 页面标题 */}
      <motion.div variants={itemVariants} className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">AI 配置</h1>
        <p className="text-muted-foreground mt-1">
          管理提供商与模型。
        </p>
      </motion.div>

      {/* ========== AI 服务管理（统一卡片式） ========== */}
      <motion.div variants={itemVariants}>
        <div className="flex items-center justify-between gap-3 mb-3 px-1 flex-wrap">
          <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            API 配置与模型
          </h3>
          <div className="flex items-center gap-2 shrink-0">
            <input
              ref={importInputRef}
              type="file"
              accept="application/json,.json"
              className="hidden"
              onChange={event => {
                const file = event.target.files?.[0];
                if (file) void handleImport(file);
              }}
            />
            <button
              type="button"
              onClick={() => importInputRef.current?.click()}
              className="flex h-9 items-center gap-1.5 rounded-xl border border-border/40 px-3 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
            >
              <Upload className="h-3.5 w-3.5" />
              导入
            </button>
            <button
              type="button"
              onClick={() => { setExportIncludeSecrets(false); setExportDialogOpen(true); }}
              disabled={configs.length === 0}
              className="flex h-9 items-center gap-1.5 rounded-xl border border-border/40 px-3 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Download className="h-3.5 w-3.5" />
              导出
            </button>
            <div
              className="flex h-9 rounded-xl border border-border/30 bg-card/50 overflow-hidden shrink-0"
              role="group"
              aria-label="AI 模型展示方式"
            >
              <button
                type="button"
                onClick={() => handleViewModeChange("list")}
                aria-pressed={viewMode === "list"}
                title="列表视图"
                className={cn(
                  "flex h-full w-10 items-center justify-center transition-colors",
                  viewMode === "list"
                    ? "bg-white/10 text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <List className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={() => handleViewModeChange("card")}
                aria-pressed={viewMode === "card"}
                title="卡片视图"
                className={cn(
                  "flex h-full w-10 items-center justify-center transition-colors",
                  viewMode === "card"
                    ? "bg-white/10 text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <LayoutGrid className="h-4 w-4" />
              </button>
            </div>
            <button
              type="button"
              onClick={() => { setEditingConfig(null); setConfigDialogOpen(true); }}
              className={cn(
                "flex h-9 items-center gap-1.5 rounded-xl px-3 text-xs font-medium",
                "border border-dashed border-border/40 hover:border-primary/50",
                "text-muted-foreground hover:text-primary",
                "transition-all duration-200"
              )}
            >
              <Plus className="h-3.5 w-3.5" />
              添加 API 配置
            </button>
          </div>
        </div>

        <div
          className={cn(
            viewMode === "card"
              ? "grid grid-cols-1 lg:grid-cols-2 gap-4 items-start"
              : "space-y-4"
          )}
        >
          {configsLoading ? (
            <div className={cn(
              "flex items-center justify-center py-8",
              viewMode === "card" && "lg:col-span-2"
            )}>
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : configs.length === 0 ? (
            <div className={cn(
              "rounded-xl border border-dashed border-border/30 py-10 text-center",
              "bg-card/30",
              viewMode === "card" && "lg:col-span-2"
            )}>
              <Settings2 className="h-8 w-8 text-muted-foreground/20 mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">还没有 API 配置</p>
              <p className="text-xs text-muted-foreground/60 mt-1">点击上方「添加 API 配置」开始</p>
            </div>
          ) : (
            configs.map(renderConfigCard)
          )}
        </div>
      </motion.div>

      {/* Dialogs */}
      <Dialog open={exportDialogOpen} onOpenChange={setExportDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>导出配置</DialogTitle>
            <DialogDescription>选择导出内容。</DialogDescription>
          </DialogHeader>
          <button
            type="button"
            onClick={() => setExportIncludeSecrets(value => !value)}
            className="flex items-center justify-between rounded-xl border border-border/40 bg-muted/15 px-3 py-3 text-left"
          >
            <span>
              <span className="block text-sm font-medium">包含密钥</span>
              <span className="mt-0.5 block text-[10px] text-muted-foreground">仅在可信环境中使用</span>
            </span>
            <span className={cn("relative h-5 w-9 rounded-full transition-colors", exportIncludeSecrets ? "bg-primary" : "bg-muted-foreground/25")}>
              <span className={cn("absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform", exportIncludeSecrets && "translate-x-4")} />
            </span>
          </button>
          <DialogFooter>
            <DialogClose render={<Button variant="outline" size="sm" />}>取消</DialogClose>
            <Button size="sm" onClick={() => handleExport(exportIncludeSecrets)}>
              <Download className="h-3.5 w-3.5" />导出
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <ApiConfigDialog
        open={configDialogOpen}
        onOpenChange={setConfigDialogOpen}
        editingConfig={editingConfig}
        onSaved={() => { loadConfigs(); loadModels(); }}
      />
      <AiModelDialog
        open={modelDialogOpen}
        onOpenChange={setModelDialogOpen}
        editingModel={editingModel}
        apiConfigs={configs}
        defaultApiConfigId={modelDialogApiConfigId}
        onSaved={() => { loadModels(); }}
      />
      {fetchModelsConfig && (
        <FetchRemoteModelsDialog
          open={fetchModelsDialogOpen}
          onOpenChange={setFetchModelsDialogOpen}
          apiConfig={fetchModelsConfig}
          existingModelCodes={new Set(models.filter(m => m.apiConfigId === fetchModelsConfig.id).map(m => m.code))}
          onAdded={() => { loadModels(); }}
        />
      )}
    </motion.div>
  );
}
