"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { Loader2, Plus, Settings2, Upload } from "lucide-react";
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
import type { ProviderModelCandidate } from "@/lib/api/provider-catalog";
import { Button } from "@/components/ui/button";
import { AudioGenerationDialog } from "@/components/generation/audio/audio-generation-dialog";
import { containerVariants, itemVariants, settingsTypography } from "../_shared";
import { useProviderCatalog } from "./_hooks/use-provider-catalog";
import { useCardMasonryLayout } from "./_hooks/use-card-masonry-layout";
import { AiModelDialog } from "./_components/ai-model-dialog";
import type { ModelConnectivityTestState } from "./_components/ai-model-row";
import { ApiConfigCard } from "./_components/api-config-card";
import { ApiConfigDialog } from "./_components/api-config-dialog";
import { ComfyUiWorkflowManagerDialog } from "./_components/comfyui-workflow-manager-dialog";
import { DefaultModelsBar } from "./_components/default-models-bar";
import { ExportConfigDialog } from "./_components/export-config-dialog";
import { FetchRemoteModelsDialog } from "./_components/fetch-remote-models-dialog";
import {
  ProviderSetupDialog,
  type ProviderSetupTarget,
} from "./_components/provider-setup-dialog";
import {
  filterApiConfigsAndModels,
  resolveDefaultModelsByType,
  type AiModelStatusFilter,
} from "./_components/ai-models-filter";
import {
  exportAiModelsConfigFile,
  importAiModelsConfigFile,
} from "./_components/ai-models-io";
import {
  AiModelsToolbar,
  type AiModelViewMode,
} from "./_components/ai-models-toolbar";

const AI_MODEL_VIEW_MODE_STORAGE_KEY = "fusion-ai-model-view-mode";

function isAiModelViewMode(value: string | null): value is AiModelViewMode {
  return value === "list" || value === "card";
}

export default function AiModelsPage() {
  const providerCatalog = useProviderCatalog();
  const { confirm } = useConfirm();
  const shouldReduceMotion = Boolean(useReducedMotion());

  const [models, setModels] = useState<AiModel[]>([]);
  const [modelPresets, setModelPresets] = useState<ModelPreset[]>([]);
  const [configs, setConfigs] = useState<ApiConfig[]>([]);
  const [configsLoading, setConfigsLoading] = useState(true);
  const [loadErrors, setLoadErrors] = useState({
    configs: false,
    models: false,
    presets: false,
  });

  // 筛选、视图与折叠状态
  const [query, setQuery] = useState("");
  const [modelTypeFilter, setModelTypeFilter] = useState(0);
  const [statusFilter, setStatusFilter] = useState<AiModelStatusFilter>("all");
  const [viewMode, setViewMode] = useState<AiModelViewMode>("list");
  const [viewModeReady, setViewModeReady] = useState(false);
  const [collapsedConfigIds, setCollapsedConfigIds] = useState<Set<number>>(new Set());
  const [expandedFilterConfigIds, setExpandedFilterConfigIds] = useState<Set<number>>(new Set());
  const [highlightedConfigId, setHighlightedConfigId] = useState<number | null>(null);

  // 对话框状态
  const [setupTarget, setSetupTarget] = useState<ProviderSetupTarget | null>(null);
  const [initialApiValues, setInitialApiValues] = useState<ApiConfigSaveReq | undefined>();
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState<ApiConfig | null>(null);
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [editingModel, setEditingModel] = useState<AiModel | null>(null);
  const [tryModel, setTryModel] = useState<AiModel | null>(null);
  const [initialCandidate, setInitialCandidate] = useState<ProviderModelCandidate | null>(null);
  const [modelDialogApiConfigId, setModelDialogApiConfigId] = useState<number | undefined>(undefined);
  const [fetchModelsDialogOpen, setFetchModelsDialogOpen] = useState(false);
  const [fetchModelsConfig, setFetchModelsConfig] = useState<ApiConfig | null>(null);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportIncludeSecrets, setExportIncludeSecrets] = useState(false);
  const [workflowManagerConfig, setWorkflowManagerConfig] = useState<ApiConfig | null>(null);

  // 运行时交互状态
  const [testingModelIds, setTestingModelIds] = useState<Set<number>>(new Set());
  const [testResults, setTestResults] = useState<Record<number, ModelConnectivityTestState>>({});
  const [settingDefaultModelId, setSettingDefaultModelId] = useState<number | null>(null);
  const [updatingDefaultType, setUpdatingDefaultType] = useState<number | null>(null);
  const [togglingConfigIds, setTogglingConfigIds] = useState<Set<number>>(new Set());
  const [togglingModelIds, setTogglingModelIds] = useState<Set<number>>(new Set());

  const importInputRef = useRef<HTMLInputElement>(null);
  const cardGridRef = useRef<HTMLDivElement>(null);
  const highlightTimerRef = useRef<number | null>(null);

  const loadModels = useCallback(async () => {
    try {
      const data = await aiModelApi.listAll();
      setModels(data);
      setLoadErrors((current) => ({ ...current, models: false }));
    } catch (err) {
      console.error("加载 AI 模型列表失败:", err);
      setLoadErrors((current) => ({ ...current, models: true }));
    }
  }, []);

  const loadConfigs = useCallback(async (options?: { showLoading?: boolean }) => {
    const showLoading = options?.showLoading === true;
    try {
      if (showLoading) {
        setConfigsLoading(true);
      }
      const data = await apiConfigApi.listAll();
      setConfigs(data);
      setLoadErrors((current) => ({ ...current, configs: false }));
    } catch (err) {
      console.error("加载 API 配置列表失败:", err);
      setLoadErrors((current) => ({ ...current, configs: true }));
    } finally {
      setConfigsLoading(false);
    }
  }, []);

  const loadModelPresets = useCallback(async () => {
    try {
      const data = await aiModelApi.presets();
      setModelPresets(data);
      setLoadErrors((current) => ({ ...current, presets: false }));
    } catch (err) {
      console.error("加载模型预设失败:", err);
      setLoadErrors((current) => ({ ...current, presets: true }));
    }
  }, []);

  useEffect(() => {
    void loadModels();
    void loadConfigs({ showLoading: true });
    void loadModelPresets();
  }, [loadConfigs, loadModelPresets, loadModels]);

  useEffect(() => {
    const storedViewMode = window.localStorage.getItem(AI_MODEL_VIEW_MODE_STORAGE_KEY);
    if (isAiModelViewMode(storedViewMode)) {
      setViewMode(storedViewMode);
    }
    setViewModeReady(true);
    return () => {
      if (highlightTimerRef.current !== null) {
        window.clearTimeout(highlightTimerRef.current);
      }
    };
  }, []);

  const hasActiveFilters =
    query.trim().length > 0 || modelTypeFilter !== 0 || statusFilter !== "all";

  const visibleEntries = useMemo(
    () =>
      filterApiConfigsAndModels(configs, models, {
        query,
        modelType: modelTypeFilter,
        status: statusFilter,
      }),
    [configs, models, query, modelTypeFilter, statusFilter],
  );

  const defaultSlots = useMemo(
    () => resolveDefaultModelsByType(models, configs),
    [models, configs],
  );

  const allVisibleCollapsed = useMemo(() => {
    if (visibleEntries.length === 0) {
      return false;
    }
    return visibleEntries.every((entry) => collapsedConfigIds.has(entry.config.id));
  }, [visibleEntries, collapsedConfigIds]);

  const layoutSignature = useMemo(
    () =>
      visibleEntries
        .map(
          (entry) =>
            `${entry.config.id}:${entry.visibleModels.length}:${expandedFilterConfigIds.has(entry.config.id) ? 1 : 0}`,
        )
        .join("|"),
    [visibleEntries, expandedFilterConfigIds],
  );

  useCardMasonryLayout(cardGridRef, {
    viewMode,
    disabled: configsLoading || !viewModeReady,
    layoutSignature,
    shouldReduceMotion,
  });

  const handleTestModel = async (model: AiModel) => {
    if (model.modelType === 4) {
      setTryModel(model);
      return;
    }
    setTestingModelIds((prev) => new Set(prev).add(model.id));
    try {
      const result = await aiModelApi.testTextConnectivity(model.id);
      const preview = result.responseText.replace(/\s+/g, " ").trim();
      const summary = preview.length > 120 ? `${preview.slice(0, 120)}...` : preview;
      const description = [
        summary ? `回复：${summary}` : "已收到模型响应",
        typeof result.durationMs === "number" ? `${result.durationMs} ms` : "",
      ]
        .filter(Boolean)
        .join(" · ");

      setTestResults((prev) => ({
        ...prev,
        [model.id]: { ok: true, durationMs: result.durationMs, summary: description },
      }));
      toast.success(`${result.modelName} 连通正常`, { description });
    } catch (err) {
      const message = err instanceof Error ? err.message : "请求失败";
      setTestResults((prev) => ({
        ...prev,
        [model.id]: { ok: false, summary: message },
      }));
      toast.error(`${getModelDisplayParts(model).name} 检测失败`, { description: message });
    } finally {
      setTestingModelIds((prev) => {
        const next = new Set(prev);
        next.delete(model.id);
        return next;
      });
    }
  };

  const handleSetDefaultModel = async (model: AiModel) => {
    setSettingDefaultModelId(model.id);
    setUpdatingDefaultType(model.modelType);
    try {
      await aiModelApi.update({ id: model.id, defaultModel: true });
      await loadModels();
      toast.success(`已将「${getModelDisplayParts(model).name}」设为默认模型`);
    } catch (err) {
      console.error("设置默认模型失败:", err);
      toastApiError(err, "设置默认模型失败");
    } finally {
      setSettingDefaultModelId(null);
      setUpdatingDefaultType(null);
    }
  };

  const handleToggleConfigStatus = async (config: ApiConfig) => {
    const nextStatus = config.status === 1 ? 0 : 1;
    setTogglingConfigIds((prev) => new Set(prev).add(config.id));
    try {
      await apiConfigApi.update({ id: config.id, name: config.name, status: nextStatus });
      await loadConfigs();
      toast.success(
        nextStatus === 1 ? `已启用「${config.name}」` : `已停用「${config.name}」`,
      );
    } catch (err) {
      console.error("切换 API 配置状态失败:", err);
      toastApiError(err, "更新服务状态失败");
    } finally {
      setTogglingConfigIds((prev) => {
        const next = new Set(prev);
        next.delete(config.id);
        return next;
      });
    }
  };

  const handleToggleModelStatus = async (model: AiModel) => {
    const nextStatus = model.status === 1 ? 0 : 1;
    setTogglingModelIds((prev) => new Set(prev).add(model.id));
    try {
      await aiModelApi.update({ id: model.id, status: nextStatus });
      await loadModels();
      const displayName = getModelDisplayParts(model).name;
      toast.success(
        nextStatus === 1 ? `已启用模型「${displayName}」` : `已停用模型「${displayName}」`,
      );
    } catch (err) {
      console.error("切换模型状态失败:", err);
      toastApiError(err, "更新模型状态失败");
    } finally {
      setTogglingModelIds((prev) => {
        const next = new Set(prev);
        next.delete(model.id);
        return next;
      });
    }
  };

  const handleDeleteModel = async (model: AiModel) => {
    const displayName = getModelDisplayParts(model).name;
    const ok = await confirm({
      title: "删除 AI 模型",
      description: model.defaultModel
        ? `确定删除默认模型「${displayName}」吗？删除后该能力类型将没有默认模型，此操作不可撤销。`
        : `确定删除模型「${displayName}」吗？此操作不可撤销。`,
      variant: "destructive",
      confirmText: "确定删除",
    });
    if (!ok) {
      return;
    }
    try {
      await aiModelApi.delete(model.id);
      await loadModels();
      toast.success(`已删除模型「${displayName}」`);
    } catch (err) {
      console.error("删除模型失败:", err);
      toastApiError(err, "删除模型失败");
    }
  };

  const handleDeleteConfig = async (config: ApiConfig) => {
    const linkedCount = models.filter((model) => model.apiConfigId === config.id).length;
    const ok = await confirm({
      title: "删除 API 配置",
      description: linkedCount > 0
        ? `确定删除「${config.name}」吗？其下关联的 ${linkedCount} 个模型将失去连接配置而无法调用，此操作不可撤销。`
        : `确定删除「${config.name}」吗？此操作不可撤销。`,
      variant: "destructive",
      confirmText: "确定删除",
    });
    if (!ok) {
      return;
    }
    try {
      await apiConfigApi.delete(config.id);
      await Promise.all([loadConfigs(), loadModels()]);
      toast.success(`已删除 API 配置「${config.name}」`);
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
    setCollapsedConfigIds((prev) => {
      const next = new Set(prev);
      if (next.has(configId)) {
        next.delete(configId);
      } else {
        next.add(configId);
      }
      return next;
    });
  };

  const handleToggleCollapseAll = () => {
    setCollapsedConfigIds((prev) => {
      const next = new Set(prev);
      for (const entry of visibleEntries) {
        if (allVisibleCollapsed) {
          next.delete(entry.config.id);
        } else {
          next.add(entry.config.id);
        }
      }
      return next;
    });
  };

  const handleLocateConfig = (configId: number) => {
    setQuery("");
    setModelTypeFilter(0);
    setStatusFilter("all");
    setCollapsedConfigIds((prev) => {
      const next = new Set(prev);
      next.delete(configId);
      return next;
    });
    setHighlightedConfigId(configId);
    if (highlightTimerRef.current !== null) {
      window.clearTimeout(highlightTimerRef.current);
    }
    highlightTimerRef.current = window.setTimeout(() => {
      setHighlightedConfigId(null);
      highlightTimerRef.current = null;
    }, 2000);
    requestAnimationFrame(() => {
      document.getElementById(`api-config-card-${configId}`)?.scrollIntoView({
        behavior: shouldReduceMotion ? "auto" : "smooth",
        block: "center",
      });
    });
  };

  const handleExport = (includeSecrets: boolean) => {
    exportAiModelsConfigFile(configs, models, includeSecrets);
    toast.success(includeSecrets ? "配置已导出（含密钥）" : "配置已导出");
    setExportDialogOpen(false);
  };

  const handleImport = async (file: File) => {
    try {
      const { configurationsAwaitingKey } = await importAiModelsConfigFile(
        file,
        providerCatalog.providers,
      );
      await Promise.all([loadConfigs(), loadModels()]);
      toast.success("配置已导入", {
        description: configurationsAwaitingKey
          ? `已保留模型配置；${configurationsAwaitingKey} 份平台配置因未包含密钥而停用，请填写密钥后启用。`
          : "已创建新的 API 配置和模型，原配置不会被覆盖。",
      });
    } catch (error) {
      toast.error("导入失败", {
        description: error instanceof Error ? error.message : "请检查 JSON 文件",
      });
    } finally {
      if (importInputRef.current) {
        importInputRef.current.value = "";
      }
    }
  };

  const openCreateConfig = () => {
    setInitialCandidate(null);
    setInitialApiValues(undefined);
    setEditingConfig(null);
    setConfigDialogOpen(true);
  };

  const resetFilters = () => {
    setQuery("");
    setModelTypeFilter(0);
    setStatusFilter("all");
    setExpandedFilterConfigIds(new Set());
  };

  const isLoadingView = configsLoading || !viewModeReady;

  return (
    <motion.div
      className="w-full"
      variants={containerVariants}
      initial={false}
      animate="visible"
    >
      <motion.div variants={itemVariants} className="mb-6">
        <h1 className={settingsTypography.pageTitle}>AI 配置</h1>
        <p className={settingsTypography.pageDescription}>
          管理模型服务商连接、能力协议与全局默认模型。
        </p>
      </motion.div>

      {!isLoadingView && !Object.values(loadErrors).some(Boolean) && configs.length > 0 && (
        <motion.div variants={itemVariants}>
          <DefaultModelsBar
            slots={defaultSlots}
            updatingModelType={updatingDefaultType}
            onSelectDefault={handleSetDefaultModel}
            onLocateConfig={handleLocateConfig}
          />
        </motion.div>
      )}

      <motion.div variants={itemVariants}>
        <AiModelsToolbar
          query={query}
          onQueryChange={setQuery}
          modelTypeFilter={modelTypeFilter}
          onModelTypeFilterChange={setModelTypeFilter}
          statusFilter={statusFilter}
          onStatusFilterChange={setStatusFilter}
          viewMode={viewMode}
          onViewModeChange={handleViewModeChange}
          hasConfigs={configs.length > 0}
          allCollapsed={allVisibleCollapsed}
          onToggleCollapseAll={handleToggleCollapseAll}
          importInputRef={importInputRef}
          onImportFile={(file) => void handleImport(file)}
          onOpenExport={() => {
            setExportIncludeSecrets(false);
            setExportDialogOpen(true);
          }}
          onOpenCreateConfig={openCreateConfig}
          totalConfigCount={configs.length}
          visibleConfigCount={visibleEntries.length}
          hasActiveFilters={hasActiveFilters}
          onResetFilters={resetFilters}
        />

        <div
          ref={cardGridRef}
          className={cn(
            viewMode === "card"
              ? "grid grid-cols-1 items-start gap-4 lg:grid-cols-2"
              : "flex flex-col gap-4",
          )}
        >
          {isLoadingView ? (
            <div
              role="status"
              aria-label="加载模型配置"
              className={cn(
                "flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground",
                viewMode === "card" && "lg:col-span-2",
              )}
            >
              <Loader2 className="size-4 animate-spin" />
              正在加载模型配置…
            </div>
          ) : Object.values(loadErrors).some(Boolean) ? (
            <div
              role="alert"
              className={cn(
                "flex flex-col items-center gap-3 rounded-xl border border-border/30 bg-card/50 py-10 text-center",
                viewMode === "card" && "lg:col-span-2",
              )}
            >
              <p className="text-sm text-muted-foreground">模型配置暂时无法加载，请重试</p>
              <Button
                variant="outline"
                onClick={() =>
                  void Promise.all([
                    loadConfigs({ showLoading: true }),
                    loadModels(),
                    loadModelPresets(),
                  ])
                }
              >
                重试
              </Button>
            </div>
          ) : configs.length === 0 ? (
            <div
              className={cn(
                "flex flex-col items-center gap-3 rounded-xl border border-dashed border-border/40 bg-card/30 px-6 py-12 text-center",
                viewMode === "card" && "lg:col-span-2",
              )}
            >
              <span className="grid size-12 place-items-center rounded-2xl bg-muted/60 text-muted-foreground">
                <Settings2 className="size-6" />
              </span>
              <div className="space-y-1">
                <p className="text-sm font-medium text-foreground">还没有 API 配置</p>
                <p className="text-xs text-muted-foreground">
                  先添加一个模型服务商连接，即可配置文本、图像、视频或音频模型。
                </p>
              </div>
              <div className="mt-1 flex flex-wrap items-center justify-center gap-2">
                <Button type="button" onClick={openCreateConfig}>
                  <Plus />
                  添加 API 配置
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => importInputRef.current?.click()}
                >
                  <Upload />
                  导入配置
                </Button>
              </div>
            </div>
          ) : visibleEntries.length === 0 ? (
            <div
              role="status"
              className={cn(
                "flex flex-col items-center gap-3 rounded-xl border border-border/30 bg-card/40 px-6 py-10 text-center",
                viewMode === "card" && "lg:col-span-2",
              )}
            >
              <p className="text-sm text-muted-foreground">
                没有匹配当前筛选条件的服务或模型。
              </p>
              <Button type="button" variant="outline" onClick={resetFilters}>
                重置筛选
              </Button>
            </div>
          ) : (
            visibleEntries.map((entry) => {
              const { config } = entry;
              const provider = providerCatalog.providers.find(
                (item) => item.id === config.providerId,
              );
              return (
                <ApiConfigCard
                  key={config.id}
                  config={config}
                  provider={provider}
                  allModels={entry.allModels}
                  visibleModels={entry.visibleModels}
                  hiddenModelCount={entry.hiddenModelCount}
                  showingAllInFilter={expandedFilterConfigIds.has(config.id)}
                  onToggleShowAllInFilter={() => {
                    setExpandedFilterConfigIds((prev) => {
                      const next = new Set(prev);
                      if (next.has(config.id)) {
                        next.delete(config.id);
                      } else {
                        next.add(config.id);
                      }
                      return next;
                    });
                  }}
                  modelPresets={modelPresets}
                  collapsed={!hasActiveFilters && collapsedConfigIds.has(config.id)}
                  highlighted={highlightedConfigId === config.id}
                  testingModelIds={testingModelIds}
                  testResults={testResults}
                  settingDefaultModelId={settingDefaultModelId}
                  togglingConfigStatus={togglingConfigIds.has(config.id)}
                  togglingModelIds={togglingModelIds}
                  onToggle={() => handleToggleConfigCollapse(config.id)}
                  onToggleConfigStatus={() => void handleToggleConfigStatus(config)}
                  onFetch={() => {
                    const quickProvider = providerCatalog.providers.find(
                      (item) => item.id === config.providerId && item.quickSetup,
                    );
                    if (quickProvider) {
                      setSetupTarget({
                        provider: quickProvider,
                        existingConfig: config,
                        configuration: {
                          id: config.id,
                          name: config.name,
                          providerId: quickProvider.id,
                        },
                      });
                    } else {
                      setFetchModelsConfig(config);
                      setFetchModelsDialogOpen(true);
                    }
                  }}
                  onWorkflow={() => setWorkflowManagerConfig(config)}
                  onEditConfig={() => {
                    setEditingConfig(config);
                    setConfigDialogOpen(true);
                  }}
                  onDeleteConfig={() => void handleDeleteConfig(config)}
                  onEditModel={(model) => {
                    setInitialCandidate(null);
                    setEditingModel(model);
                    setModelDialogApiConfigId(undefined);
                    setModelDialogOpen(true);
                  }}
                  onDeleteModel={(model) => void handleDeleteModel(model)}
                  onAddModelManual={() => {
                    setInitialCandidate(null);
                    setEditingModel(null);
                    setModelDialogApiConfigId(config.id);
                    setModelDialogOpen(true);
                  }}
                  onSetDefaultModel={(model) => void handleSetDefaultModel(model)}
                  onToggleModelStatus={(model) => void handleToggleModelStatus(model)}
                  onTestModel={(model) => void handleTestModel(model)}
                />
              );
            })
          )}
        </div>
      </motion.div>

      <ExportConfigDialog
        open={exportDialogOpen}
        onOpenChange={setExportDialogOpen}
        includeSecrets={exportIncludeSecrets}
        onIncludeSecretsChange={setExportIncludeSecrets}
        onExport={handleExport}
      />

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
        onCreateModel={(id) => {
          setEditingModel(null);
          setModelDialogApiConfigId(id);
          setModelDialogOpen(true);
        }}
        onSaved={async () => {
          await Promise.all([loadConfigs(), loadModels()]);
        }}
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
        onOpenChange={(open) => {
          if (!open) {
            setWorkflowManagerConfig(null);
          }
        }}
        apiConfig={workflowManagerConfig}
        onChanged={() => {
          void loadModels();
        }}
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
        onSaved={() => {
          void loadModels();
        }}
      />

      {tryModel && (
        <AudioGenerationDialog model={tryModel} onClose={() => setTryModel(null)} />
      )}

      {fetchModelsConfig && (
        <FetchRemoteModelsDialog
          open={fetchModelsDialogOpen}
          onOpenChange={setFetchModelsDialogOpen}
          apiConfig={fetchModelsConfig}
          existingModelCodes={
            new Set(
              models
                .filter((model) => model.apiConfigId === fetchModelsConfig.id)
                .map((model) => model.code),
            )
          }
          onAdded={() => {
            void loadModels();
          }}
        />
      )}
    </motion.div>
  );
}
