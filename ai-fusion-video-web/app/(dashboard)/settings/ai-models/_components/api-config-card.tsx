"use client";

import {
  ChevronRight,
  CloudDownload,
  Edit2,
  KeyRound,
  Plus,
  Trash2,
  Workflow,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  MODEL_TYPE_LABELS,
  PLATFORM_LABELS,
  type AiModel,
  type ApiConfig,
  type ModelPreset,
} from "@/lib/api/ai-model";
import type { ProviderDefinition } from "@/lib/api/provider-catalog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { ProviderVendorIcon } from "@/components/dashboard/model-vendor-icon";
import { maskSecret } from "../../_shared";
import { SponsorNotice } from "./sponsor-notice";
import { formatSemanticLabel, MODEL_PROTOCOL_LABELS } from "./model-config-support";
import {
  AiModelRow,
  type ModelConnectivityTestState,
} from "./ai-model-row";

interface ApiConfigCardProps {
  config: ApiConfig;
  provider?: ProviderDefinition;
  allModels: AiModel[];
  visibleModels: AiModel[];
  hiddenModelCount: number;
  showingAllInFilter: boolean;
  onToggleShowAllInFilter: () => void;
  modelPresets: ModelPreset[];
  collapsed: boolean;
  highlighted: boolean;
  testingModelIds: Set<number>;
  testResults: Record<number, ModelConnectivityTestState>;
  settingDefaultModelId: number | null;
  togglingConfigStatus: boolean;
  togglingModelIds: Set<number>;
  onToggle: () => void;
  onToggleConfigStatus: () => void;
  onFetch: () => void;
  onWorkflow: () => void;
  onEditConfig: () => void;
  onDeleteConfig: () => void;
  onEditModel: (model: AiModel) => void;
  onDeleteModel: (model: AiModel) => void;
  onAddModelManual: () => void;
  onSetDefaultModel: (model: AiModel) => void;
  onToggleModelStatus: (model: AiModel) => void;
  onTestModel: (model: AiModel) => void;
}

export function ApiConfigCard({
  config,
  provider,
  allModels,
  visibleModels,
  hiddenModelCount,
  showingAllInFilter,
  onToggleShowAllInFilter,
  modelPresets,
  collapsed,
  highlighted,
  testingModelIds,
  testResults,
  settingDefaultModelId,
  togglingConfigStatus,
  togglingModelIds,
  onToggle,
  onToggleConfigStatus,
  onFetch,
  onWorkflow,
  onEditConfig,
  onDeleteConfig,
  onEditModel,
  onDeleteModel,
  onAddModelManual,
  onSetDefaultModel,
  onToggleModelStatus,
  onTestModel,
}: ApiConfigCardProps) {
  const shouldReduceMotion = useReducedMotion();
  const displayModels = showingAllInFilter ? allModels : visibleModels;
  const isDisabled = config.status === 0;
  const awaitingKey = Boolean(provider?.quickSetup && isDisabled && !config.apiKey?.trim());
  const supportsRemoteList = config.platform !== "comfyui" && config.platform !== "sonilo";

  const modelTypeGroups = [1, 2, 3, 4]
    .map((type) => ({
      type,
      label: MODEL_TYPE_LABELS[type],
      models: displayModels.filter((model) => model.modelType === type),
    }))
    .filter((group) => group.models.length > 0);

  const defaultProtocols = (
    [
      { label: "文", protocol: config.textProtocol },
      { label: "图", protocol: config.imageProtocol },
      { label: "视", protocol: config.videoProtocol },
    ] as const
  ).filter((item) => Boolean(item.protocol));

  return (
    <section
      id={`api-config-card-${config.id}`}
      data-api-config-card
      className={cn(
        "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm transition-shadow",
        highlighted && "border-primary/50 ring-2 ring-primary/35",
      )}
    >
      {/* API 配置头部 */}
      <div
        className={cn(
          "flex flex-col gap-3 rounded-t-xl bg-muted/50 px-4 py-3 sm:flex-row sm:items-start",
          collapsed && "rounded-b-xl",
          isDisabled && "bg-muted/30",
        )}
      >
        <div className="flex min-w-0 flex-1 items-start gap-3">
          <button
            type="button"
            onClick={onToggle}
            aria-expanded={!collapsed}
            aria-label={collapsed ? `展开 ${config.name}` : `折叠 ${config.name}`}
            title={collapsed ? "展开配置" : "折叠配置"}
            className="mt-1 flex size-7 shrink-0 items-center justify-center rounded-lg text-muted-foreground outline-none transition-colors hover:bg-muted/60 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
          >
            <ChevronRight
              className={cn(
                "size-4 transition-transform duration-200 motion-reduce:transition-none",
                !collapsed && "rotate-90",
              )}
            />
          </button>

          <div className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted/60">
            <ProviderVendorIcon
              provider={provider?.vendor ?? config.providerId ?? config.platform}
              className="size-5"
            />
          </div>

          <div className="min-w-0 flex-1 space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <h4
                className={cn(
                  "text-sm font-semibold break-words",
                  isDisabled ? "text-muted-foreground" : "text-foreground",
                )}
              >
                {config.name}
              </h4>
              {provider?.sponsored && (
                <Badge
                  variant="secondary"
                  className="bg-primary/10 text-primary"
                >
                  赞助合作伙伴
                </Badge>
              )}
              {config.platform && (
                <Badge variant="secondary">
                  {PLATFORM_LABELS[config.platform] || config.platform}
                </Badge>
              )}
              {isDisabled && (
                <Badge variant="outline" className="text-muted-foreground">
                  已停用
                </Badge>
              )}
            </div>

            {provider?.sponsored && <SponsorNotice provider={provider} />}

            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
              {config.apiKey && (
                <span className="font-mono" title="已保存的密钥（脱敏展示）">
                  {maskSecret(config.apiKey)}
                </span>
              )}
              {config.proxyType
                && config.proxyType !== "none"
                && config.proxyHost
                && config.proxyPort && (
                  <span className="rounded-md bg-sky-500/10 px-1.5 py-0.5 font-mono text-sky-700 break-all dark:text-sky-400">
                    {config.proxyType}://{config.proxyHost}:{config.proxyPort}
                    {config.proxyUsername ? " auth" : ""}
                  </span>
                )}
              <span className="tabular-nums">{allModels.length} 个模型</span>
              {defaultProtocols.map((item) => (
                <span
                  key={item.label}
                  className="rounded-md bg-violet-500/10 px-1.5 py-0.5 text-violet-700 dark:text-violet-400"
                  title={`${item.label}类默认协议：${formatSemanticLabel(item.protocol, MODEL_PROTOCOL_LABELS)}`}
                >
                  {item.label} · {formatSemanticLabel(item.protocol, MODEL_PROTOCOL_LABELS)}
                </span>
              ))}
            </div>

            {awaitingKey && (
              <div className="flex flex-wrap items-center gap-2 pt-1 text-xs text-amber-700 dark:text-amber-400">
                <span className="inline-flex items-center gap-1">
                  <KeyRound className="size-3.5" />
                  导入时未包含密钥，请补充密钥后启用服务。
                </span>
                <Button
                  type="button"
                  variant="outline"
                  size="xs"
                  onClick={onEditConfig}
                >
                  补充密钥
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* 右侧操作区：全部 32px 对齐 */}
        <div className="flex flex-wrap items-center gap-1.5 sm:shrink-0 sm:pt-0.5">
          <div className="flex h-8 items-center px-1">
            <Switch
              checked={!isDisabled}
              disabled={togglingConfigStatus}
              aria-label={isDisabled ? `启用服务 ${config.name}` : `停用服务 ${config.name}`}
              title={
                awaitingKey
                  ? "填写密钥后可启用服务"
                  : isDisabled
                    ? "启用服务"
                    : "停用服务"
              }
              onCheckedChange={() => {
                if (awaitingKey) {
                  onEditConfig();
                  return;
                }
                onToggleConfigStatus();
              }}
            />
          </div>

          {config.platform === "comfyui" && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onWorkflow}
              aria-label={`管理 ${config.name} 的 ComfyUI 工作流`}
              title="管理 ComfyUI 工作流"
            >
              <Workflow />
              管理工作流
            </Button>
          )}

          <div className="flex items-center gap-1 text-muted-foreground">
            {supportsRemoteList && (
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={onFetch}
                aria-label={
                  provider?.quickSetup
                    ? `从 ${config.name} 快速勾选模型`
                    : `获取 ${config.name} 的可用模型列表`
                }
                title={provider?.quickSetup ? "快速勾选平台模型" : "获取可用模型列表"}
              >
                <CloudDownload />
              </Button>
            )}

            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              onClick={onEditConfig}
              aria-label={`编辑 API 配置 ${config.name}`}
              title="编辑 API 配置"
            >
              <Edit2 />
            </Button>

            <Button
              type="button"
              variant="destructive-ghost"
              size="icon-sm"
              onClick={onDeleteConfig}
              aria-label={`删除 API 配置 ${config.name}`}
              title="删除 API 配置"
            >
              <Trash2 />
            </Button>
          </div>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {!collapsed && (
          <motion.div
            key="config-models"
            initial={shouldReduceMotion ? false : { height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={
              shouldReduceMotion
                ? { duration: 0 }
                : { duration: 0.22, ease: [0.25, 0.46, 0.45, 0.94] }
            }
            className="overflow-hidden border-t border-border/20"
          >
            <div className="flex flex-col gap-2 px-3 py-3">
              {modelTypeGroups.length === 0 ? (
                <p className="py-4 text-center text-xs text-muted-foreground">
                  {allModels.length === 0 ? "尚未添加模型" : "当前筛选条件下暂无匹配模型"}
                </p>
              ) : (
                modelTypeGroups.map((group, groupIndex) => (
                  <div
                    key={group.type}
                    className={cn(
                      groupIndex > 0 && "mt-1 border-t border-border/20 pt-2",
                    )}
                  >
                    <div className="flex items-center gap-2 px-3 py-1">
                      <span className="shrink-0 text-xs font-medium text-muted-foreground">
                        {group.label}
                      </span>
                      <span className="text-xs text-muted-foreground/70 tabular-nums">
                        {group.models.length}
                      </span>
                      <div className="h-px flex-1 bg-border/20" />
                    </div>

                    <div className="flex flex-col gap-0.5">
                      {group.models.map((model) => (
                        <AiModelRow
                          key={model.id}
                          model={model}
                          config={config}
                          modelPresets={modelPresets}
                          testing={testingModelIds.has(model.id)}
                          testResult={testResults[model.id]}
                          settingDefault={settingDefaultModelId === model.id}
                          togglingStatus={togglingModelIds.has(model.id)}
                          onTestModel={onTestModel}
                          onSetDefault={onSetDefaultModel}
                          onToggleStatus={onToggleModelStatus}
                          onEditModel={onEditModel}
                          onDeleteModel={onDeleteModel}
                        />
                      ))}
                    </div>
                  </div>
                ))
              )}

              {hiddenModelCount > 0 && (
                <div className="flex items-center justify-between gap-2 rounded-lg border border-border/20 bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
                  <span>
                    {showingAllInFilter
                      ? `已展示该服务下全部 ${allModels.length} 个模型`
                      : `已隐藏该服务下其余 ${hiddenModelCount} 个未匹配模型`}
                  </span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="xs"
                    onClick={onToggleShowAllInFilter}
                  >
                    {showingAllInFilter ? "仅看匹配项" : "显示全部"}
                  </Button>
                </div>
              )}

              {/* 底部模型添加区：区分平台勾选与手动添加 */}
              <div className="flex flex-wrap items-center gap-2 border-t border-border/20 pt-2.5">
                {provider?.quickSetup ? (
                  <>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={onFetch}
                    >
                      <CloudDownload />
                      从平台勾选模型
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={onAddModelManual}
                    >
                      <Plus />
                      手动添加模型
                    </Button>
                  </>
                ) : (
                  <>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={onAddModelManual}
                    >
                      <Plus />
                      添加模型
                    </Button>
                    {supportsRemoteList && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={onFetch}
                      >
                        <CloudDownload />
                        从远程拉取
                      </Button>
                    )}
                  </>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
