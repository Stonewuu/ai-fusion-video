"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  Download,
  LayoutGrid,
  List,
  Loader2,
  Plus,
  Settings2,
  Upload,
} from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/utils";
import { toastApiError } from "@/lib/api/toast-api-error";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { getModelDisplayParts } from "@/lib/model-display";
import {
  aiModelApi,
  apiConfigApi,
  type AiModel,
  type ApiConfig,
  type ApiConfigSaveReq,
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
  containerVariants,
  itemVariants,
  settingsTypography,
} from "../_shared";
import { AiModelDialog } from "./_components/ai-model-dialog";
import { ApiConfigCard } from "./_components/api-config-card";
import { ApiConfigDialog } from "./_components/api-config-dialog";
import { ProviderSetupDialog, type ProviderSetupTarget } from "./_components/provider-setup-dialog";
import type { ProviderModelCandidate } from "@/lib/api/provider-catalog";
import { useProviderCatalog } from "./_hooks/use-provider-catalog";
import { ApiConfigSearch, matchingApiConfigs } from "./_components/api-config-search";
import { FetchRemoteModelsDialog } from "./_components/fetch-remote-models-dialog";
import { ComfyUiWorkflowManagerDialog } from "./_components/comfyui-workflow-manager-dialog";

type AiModelViewMode = "list" | "card";

const AI_MODEL_VIEW_MODE_STORAGE_KEY = "fusion-ai-model-view-mode";

function isAiModelViewMode(value: string | null): value is AiModelViewMode {
  return value === "list" || value === "card";
}

export default function AiModelsPage() {
  const providerCatalog = useProviderCatalog();
  const [setupTarget, setSetupTarget] = useState<ProviderSetupTarget | null>(null);
  const [initialApiValues, setInitialApiValues] = useState<ApiConfigSaveReq | undefined>();
  const { confirm } = useConfirm();
  const shouldReduceMotion = useReducedMotion();

  // AI 模型列表
  const [models, setModels] = useState<AiModel[]>([]);
  const [modelPresets, setModelPresets] = useState<ModelPreset[]>([]);

  // API 配置列表
  const [configs, setConfigs] = useState<ApiConfig[]>([]);
  const [query, setQuery] = useState("");
  const visibleConfigs = matchingApiConfigs(configs, models, query);
  const [configsLoading, setConfigsLoading] = useState(true);

  // Dialog 状态
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState<ApiConfig | null>(null);
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [editingModel, setEditingModel] = useState<AiModel | null>(null);
  const [initialCandidate, setInitialCandidate] = useState<ProviderModelCandidate | null>(null);
  const [modelDialogApiConfigId, setModelDialogApiConfigId] = useState<number | undefined>(undefined);
  const [fetchModelsDialogOpen, setFetchModelsDialogOpen] = useState(false);
  const [fetchModelsConfig, setFetchModelsConfig] = useState<ApiConfig | null>(null);
  const [testingModelIds, setTestingModelIds] = useState<Set<number>>(new Set());
  const [viewMode, setViewMode] = useState<AiModelViewMode>("list");
  const [collapsedConfigIds, setCollapsedConfigIds] = useState<Set<number>>(new Set());
  const importInputRef = useRef<HTMLInputElement>(null);
  const cardGridRef = useRef<HTMLDivElement>(null);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportIncludeSecrets, setExportIncludeSecrets] = useState(false);
  const [workflowManagerConfig, setWorkflowManagerConfig] = useState<ApiConfig | null>(null);

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
      const data = await apiConfigApi.listAll();
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

  useLayoutEffect(() => {
    const grid = cardGridRef.current;
    if (!grid) return;

    const cards = Array.from(grid.querySelectorAll<HTMLElement>("[data-api-config-card]"));
    const desktopColumns = window.matchMedia("(min-width: 1024px)");
    let layoutFrame: number | undefined;

    const resetLayout = () => {
      if (layoutFrame !== undefined) window.cancelAnimationFrame(layoutFrame);
      grid.style.gridAutoRows = "";
      cards.forEach(card => {
        card.getAnimations().forEach(animation => animation.cancel());
        card.style.gridRowEnd = "";
      });
    };

    const layoutCard = (card: HTMLElement) => {
      const gridStyle = window.getComputedStyle(grid);
      const rowHeight = Number.parseFloat(gridStyle.gridAutoRows);
      const rowGap = Number.parseFloat(gridStyle.rowGap);
      if (!Number.isFinite(rowHeight) || !Number.isFinite(rowGap)) return;

      const cardHeight = card.getBoundingClientRect().height;
      const rowSpan = Math.ceil((cardHeight + rowGap) / (rowHeight + rowGap));
      card.style.gridRowEnd = `span ${rowSpan}`;
    };

    const layoutCards = (animate: boolean) => {
      if (viewMode !== "card" || configsLoading || cards.length === 0 || !desktopColumns.matches) {
        resetLayout();
        return;
      }

      const previousRects = animate && !shouldReduceMotion
        ? new Map(cards.map(card => [card, card.getBoundingClientRect()]))
        : null;

      cards.forEach(card => {
        card.getAnimations().forEach(animation => animation.cancel());
      });
      grid.style.gridAutoRows = "1px";
      cards.forEach(layoutCard);

      if (!previousRects) return;
      cards.forEach(card => {
        const previousRect = previousRects.get(card);
        if (!previousRect) return;

        const nextRect = card.getBoundingClientRect();
        const deltaX = previousRect.left - nextRect.left;
        const deltaY = previousRect.top - nextRect.top;
        if (Math.abs(deltaX) < 1 && Math.abs(deltaY) < 1) return;

        card.animate(
          [
            { transform: `translate(${deltaX}px, ${deltaY}px)` },
            { transform: "translate(0, 0)" },
          ],
          {
            duration: 220,
            easing: "cubic-bezier(0.25, 0.46, 0.45, 0.94)",
          }
        );
      });
    };

    const scheduleAnimatedLayout = () => {
      if (layoutFrame !== undefined) window.cancelAnimationFrame(layoutFrame);
      layoutFrame = window.requestAnimationFrame(() => {
        layoutFrame = undefined;
        layoutCards(true);
      });
    };

    const resizeObserver = new ResizeObserver(() => {
      if (viewMode !== "card" || !desktopColumns.matches) return;
      scheduleAnimatedLayout();
    });

    cards.forEach(card => resizeObserver.observe(card));
    const handleColumnsChange = () => layoutCards(false);
    desktopColumns.addEventListener("change", handleColumnsChange);
    layoutCards(false);

    return () => {
      resizeObserver.disconnect();
      desktopColumns.removeEventListener("change", handleColumnsChange);
      resetLayout();
    };
  }, [configs, configsLoading, shouldReduceMotion, viewMode]);

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
    const ok = await confirm({ title: "删除 AI 模型", description: "确定要删除该 AI 模型吗？", variant: "destructive", confirmText: "确定删除" }); if (!ok) return;
    try {
      await aiModelApi.delete(id);
      await loadModels();
    } catch (err) {
      console.error("删除模型失败:", err);
      toastApiError(err, "删除模型失败");
    }
  };

  const handleDeleteConfig = async (id: number) => {
    const ok = await confirm({ title: "删除 API 配置", description: "确定要删除该 API 配置吗？", variant: "destructive", confirmText: "确定删除" }); if (!ok) return;
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
      let configurationsAwaitingKey = 0;
      for (const source of parsed.apiConfigs) {
        const requiresKey = providerCatalog.providers.find((provider) => provider.id === source.providerId)?.quickSetup;
        const awaitingKey = requiresKey && !source.apiKey?.trim();
        if (awaitingKey) {
          configurationsAwaitingKey++;
        }
        const createdId = await apiConfigApi.create({
          name: source.name,
          providerId: source.providerId || undefined,
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
          status: awaitingKey ? 0 : source.status,
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
      toast.success("配置已导入", {
        description: configurationsAwaitingKey
          ? `已保留模型配置；${configurationsAwaitingKey} 份平台配置因未包含密钥而停用，请填写密钥后启用。`
          : "已创建新的 API 配置和模型，原配置不会被覆盖。",
      });
    } catch (error) {
      toast.error("导入失败", { description: error instanceof Error ? error.message : "请检查 JSON 文件" });
    } finally {
      if (importInputRef.current) importInputRef.current.value = "";
    }
  };

  const renderConfigCard = (config: ApiConfig) => (
    <ApiConfigCard key={config.id} config={config} models={models} modelPresets={modelPresets}
      collapsed={!query.trim() && collapsedConfigIds.has(config.id)} testingModelIds={testingModelIds}
      onToggle={() => handleToggleConfigCollapse(config.id)}
      provider={providerCatalog.providers.find((provider) => provider.id === config.providerId)}
      onFetch={() => {
        const provider = providerCatalog.providers.find((item) => item.id === config.providerId && item.quickSetup);
        if (provider) {
          setSetupTarget({
            provider,
            existingConfig: config,
            configuration: { id: config.id, name: config.name, providerId: provider.id },
          });
        } else {
          setFetchModelsConfig(config);
          setFetchModelsDialogOpen(true);
        }
      }}
      onWorkflow={() => setWorkflowManagerConfig(config)}
      onEditConfig={() => { setEditingConfig(config); setConfigDialogOpen(true); }}
      onDeleteConfig={() => void handleDeleteConfig(config.id)}
      onEditModel={(model) => {
        setInitialCandidate(null);
        setEditingModel(model);
        setModelDialogApiConfigId(undefined);
        setModelDialogOpen(true);
      }}
      onDeleteModel={(id) => void handleDeleteModel(id)}
      onAddModel={() => {
        const provider = providerCatalog.providers.find((item) => item.id === config.providerId && item.quickSetup);
        if (provider) {
          setSetupTarget({
            provider,
            existingConfig: config,
            configuration: { id: config.id, name: config.name, providerId: provider.id },
          });
        } else {
          setInitialCandidate(null);
          setEditingModel(null);
          setModelDialogApiConfigId(config.id);
          setModelDialogOpen(true);
        }
      }}
      onTestTextModel={handleTestTextModel} onModelsChanged={loadModels} />
  );

  return (
    <motion.div
      className="w-full"
      variants={containerVariants}
      initial={false}
      animate="visible"
    >
      {/* 页面标题 */}
      <motion.div variants={itemVariants} className="mb-8">
        <h1 className={settingsTypography.pageTitle}>AI 配置</h1>
        <p className={settingsTypography.pageDescription}>
          管理提供商与模型。
        </p>
      </motion.div>

      {/* ========== AI 服务管理（统一卡片式） ========== */}
      <motion.div variants={itemVariants}>
        <div className="flex items-center justify-between gap-3 mb-3 px-1 flex-wrap">
          <h3 className={settingsTypography.sectionTitle}>
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
            <Button
              type="button"
              onClick={() => {
                setInitialCandidate(null);
                setInitialApiValues(undefined);
                setEditingConfig(null);
                setConfigDialogOpen(true);
              }}
            >
              <Plus />
              添加 API 配置
            </Button>
          </div>
        </div>

        <ApiConfigSearch query={query} onChange={setQuery} />
        <div
          ref={cardGridRef}
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
          ) : visibleConfigs.length === 0 ? (
            <p className="p-4 text-sm text-muted-foreground" role="status">没有匹配的服务或模型，试试其他名称。</p>
          ) : (
            visibleConfigs.map(renderConfigCard)
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
        providers={providerCatalog.providers}
        providersLoading={providerCatalog.loading}
        providersError={providerCatalog.error}
        onReloadProviders={providerCatalog.reload}
        initialValues={initialApiValues}
        onQuickSetup={(provider, configuration) => {
          setInitialCandidate(null);
          setSetupTarget({ provider, configuration });
        }}
        onCreateModel={(id) => { setEditingModel(null); setModelDialogApiConfigId(id); setModelDialogOpen(true); }}
        onSaved={async () => { await Promise.all([loadConfigs(), loadModels()]); }}
      />
      {setupTarget && (
        <ProviderSetupDialog
          target={setupTarget}
          onOpenChange={(open) => {
            if (!open) {
              setSetupTarget(null);
            }
          }}
          models={models}
          onManual={(initial, existing, candidate) => {
            setInitialCandidate(candidate ?? null);
            if (existing && candidate) {
              setEditingModel(null);
              setModelDialogApiConfigId(existing.id);
              setModelDialogOpen(true);
              return;
            }
            setInitialApiValues(initial);
            setEditingConfig(existing ?? null);
            setConfigDialogOpen(true);
          }}
          onSaved={async () => {
            await Promise.all([loadConfigs(), loadModels()]);
          }}
        />
      )}
      <ComfyUiWorkflowManagerDialog
        open={workflowManagerConfig !== null}
        onOpenChange={open => { if (!open) setWorkflowManagerConfig(null); }}
        apiConfig={workflowManagerConfig}
        onChanged={() => { loadModels(); }}
      />
      <AiModelDialog
        open={modelDialogOpen}
        onOpenChange={(open) => {
          setModelDialogOpen(open);
          if (!open) {
            setInitialCandidate(null);
          }
        }}
        editingModel={editingModel}
        apiConfigs={configs}
        providers={providerCatalog.providers}
        defaultApiConfigId={modelDialogApiConfigId}
        initialCandidate={initialCandidate}
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
