"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Check, Loader2, Search, X } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { SegmentedControl } from "@/components/shared/segmented-control";
import { toastApiError } from "@/lib/api/toast-api-error";
import { getModelDisplayParts } from "@/lib/model-display";
import { useUnsavedDialog } from "@/components/settings/use-unsaved-dialog";
import {
  aiModelApi,
  apiConfigApi,
  MODEL_TYPE_LABELS,
  MODEL_TYPE_OPTIONS,
  type ApiConfig,
  type RemoteModel,
} from "@/lib/api/ai-model";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import { Switch } from "@/components/ui/switch";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { ModelVendorIcon } from "@/components/dashboard/model-vendor-icon";
import { filterRemoteModels } from "./ai-models-filter";
import {
  formatSemanticLabel,
  MODEL_PROTOCOL_LABELS,
} from "./model-config-support";

export interface FetchRemoteModelsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  apiConfig: ApiConfig;
  existingModelCodes: Set<string>;
  onAdded: () => void;
}

export function FetchRemoteModelsDialog({
  open,
  onOpenChange,
  apiConfig,
  existingModelCodes,
  onAdded,
}: FetchRemoteModelsDialogProps) {
  const [loading, setLoading] = useState(false);
  const [remoteModels, setRemoteModels] = useState<RemoteModel[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [adding, setAdding] = useState(false);
  const [addProgress, setAddProgress] = useState<{ current: number; total: number } | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState(0);
  const [hideExisting, setHideExisting] = useState(false);
  const [defaultModelType, setDefaultModelType] = useState<number>(1);

  const changeOpen = useUnsavedDialog(
    open,
    selectedIds.size > 0,
    adding,
    onOpenChange,
  );

  const hasUnknownModelTypes = remoteModels.some((model) => model.modelType == null);
  const availableTypes = useMemo(
    () =>
      [1, 2, 3, 4].filter((type) =>
        remoteModels.some((model) => model.modelType === type),
      ),
    [remoteModels],
  );

  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const models = await apiConfigApi.remoteModels(apiConfig.id);
      setRemoteModels(models);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "获取模型列表失败";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [apiConfig.id]);

  useEffect(() => {
    if (!open) {
      return;
    }
    setRemoteModels([]);
    setError(null);
    setSelectedIds(new Set());
    setSearchQuery("");
    setTypeFilter(0);
    setHideExisting(false);
    setAddProgress(null);
    void fetchModels();
  }, [fetchModels, open]);

  const filteredModels = useMemo(
    () =>
      filterRemoteModels(remoteModels, existingModelCodes, {
        query: searchQuery,
        modelType: typeFilter,
        hideExisting,
      }),
    [remoteModels, existingModelCodes, searchQuery, typeFilter, hideExisting],
  );

  const selectableModels = filteredModels.filter(
    (model) => !existingModelCodes.has(model.id),
  );
  const allSelectableSelected =
    selectableModels.length > 0
    && selectableModels.every((model) => selectedIds.has(model.id));

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleSelectAll = () => {
    const next = new Set(selectedIds);
    for (const model of selectableModels) {
      if (allSelectableSelected) {
        next.delete(model.id);
      } else {
        next.add(model.id);
      }
    }
    setSelectedIds(next);
  };

  const handleAdd = async () => {
    if (selectedIds.size === 0) {
      return;
    }
    setAdding(true);
    const idsToAdd = Array.from(selectedIds);
    setAddProgress({ current: 0, total: idsToAdd.length });

    let succeeded = 0;
    const failedIds: string[] = [];
    let lastError: unknown = null;

    try {
      const presets = await aiModelApi.presets();
      for (const [index, modelId] of idsToAdd.entries()) {
        setAddProgress({ current: index + 1, total: idsToAdd.length });
        const remoteModel = remoteModels.find((model) => model.id === modelId);
        const preset = remoteModel?.capabilityPresetCode
          ? presets.find((item) => item.code === remoteModel.capabilityPresetCode)
          : undefined;
        try {
          await aiModelApi.create({
            name: remoteModel?.displayName || modelId,
            code: modelId,
            capabilityPresetCode: remoteModel?.capabilityPresetCode || undefined,
            modelProtocol:
              remoteModel?.modelProtocol || preset?.modelProtocol || undefined,
            modelType: remoteModel?.modelType ?? defaultModelType,
            description: preset?.description,
            supportVision:
              preset?.multimodalInputTypes?.includes("image") ?? false,
            multimodalInputTypes: preset?.multimodalInputTypes ?? [],
            multimodalInputTransports: preset?.multimodalInputTransports ?? {},
            supportReasoning: preset?.supportReasoning ?? false,
            reasoningEffortLevels: preset?.reasoningEffortLevels ?? [],
            contextWindow: preset?.contextWindow,
            apiConfigId: apiConfig.id,
          });
          succeeded++;
        } catch (err) {
          failedIds.push(modelId);
          lastError = err;
        }
      }

      if (succeeded > 0) {
        onAdded();
      }
      if (failedIds.length === 0) {
        toast.success(`已添加 ${succeeded} 个模型`);
        onOpenChange(false);
      } else if (succeeded > 0) {
        setSelectedIds(new Set(failedIds));
        toast.warning(`已添加 ${succeeded} 个模型，${failedIds.length} 个失败`, {
          description:
            lastError instanceof Error ? lastError.message : "请检查失败项配置后重试",
        });
      } else {
        toastApiError(lastError, "添加模型失败");
      }
    } catch (err) {
      console.error("添加模型失败:", err);
      toastApiError(err, "添加模型失败");
    } finally {
      setAdding(false);
      setAddProgress(null);
    }
  };

  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogContent className="flex max-h-[calc(100vh-2rem)] w-[calc(100%-2rem)] flex-col overflow-hidden sm:max-h-[88vh] sm:max-w-4xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>获取远程模型</DialogTitle>
          <DialogDescription>
            从「{apiConfig.name}」读取可用模型并批量添加到当前服务。
          </DialogDescription>
        </DialogHeader>

        {/* 固定顶栏：搜索、类型默认与筛选控件 */}
        <div className="flex shrink-0 flex-col gap-3 p-1">
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative min-w-[200px] flex-1">
              <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                aria-label="搜索远程模型"
                placeholder="搜索模型名称、ID 或归属方"
                value={searchQuery}
                disabled={adding}
                onChange={(event) => setSearchQuery(event.target.value)}
                className={cn("pl-9 text-sm", searchQuery && "pr-9")}
              />
              {searchQuery && (
                <button
                  type="button"
                  aria-label="清空搜索"
                  title="清空搜索"
                  onClick={() => setSearchQuery("")}
                  className="absolute right-2.5 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <X className="size-3.5" />
                </button>
              )}
            </div>

            {hasUnknownModelTypes && (
              <Select
                value={defaultModelType}
                disabled={adding}
                onValueChange={(value) => setDefaultModelType(value as number)}
                items={MODEL_TYPE_OPTIONS.map((option) => ({
                  value: option.value,
                  label: `未标注按：${option.label}`,
                }))}
              >
                <SelectTrigger aria-label="未标注模型的默认类型" className="w-44 text-xs">
                  <SelectValue placeholder="默认模型类型" />
                </SelectTrigger>
                <SelectContent className="text-xs">
                  <SelectGroup>
                    {MODEL_TYPE_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value} className="text-xs">
                        未标注按：{opt.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            )}
          </div>

          {!loading && !error && remoteModels.length > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                {availableTypes.length > 1 && (
                  <SegmentedControl
                    label="按类型筛选远程模型"
                    value={typeFilter}
                    onChange={setTypeFilter}
                    options={[0, ...availableTypes].map((value) => ({
                      value,
                      label: value === 0 ? "全部" : (MODEL_TYPE_LABELS[value] || `类型${value}`).replace("生成", ""),
                      disabled: adding,
                    }))}
                  />
                )}

                <div className="flex items-center gap-2 px-1">
                  <Switch
                    id="remote-hide-existing"
                    checked={hideExisting}
                    disabled={adding}
                    onCheckedChange={setHideExisting}
                  />
                  <Label
                    htmlFor="remote-hide-existing"
                    className="cursor-pointer text-xs text-muted-foreground"
                  >
                    隐藏已添加
                  </Label>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground tabular-nums">
                  显示 {filteredModels.length} 个 · 已选 {selectedIds.size} 个
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={toggleSelectAll}
                  disabled={adding || selectableModels.length === 0}
                >
                  {allSelectableSelected ? "取消全选" : "全选当前可添加"}
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* 单层列表滚动区域 */}
        <DialogScrollBody className="flex-1 p-1">
          {loading ? (
            <div
              role="status"
              className="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground"
            >
              <Loader2 className="size-4 animate-spin" />
              正在获取远程模型列表…
            </div>
          ) : error ? (
            <div role="alert" className="flex flex-col items-center gap-3 py-10 text-center">
              <p className="text-sm text-destructive">{error}</p>
              <Button variant="outline" onClick={() => void fetchModels()}>
                重试
              </Button>
            </div>
          ) : remoteModels.length === 0 ? (
            <div className="py-10 text-center text-sm text-muted-foreground">
              该服务未返回可用模型，请尝试手动添加模型。
            </div>
          ) : filteredModels.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-10 text-center">
              <p className="text-sm text-muted-foreground">没有匹配的模型，换个关键词或筛选条件试试。</p>
              {(searchQuery || typeFilter !== 0 || hideExisting) && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setSearchQuery("");
                    setTypeFilter(0);
                    setHideExisting(false);
                  }}
                >
                  清空筛选
                </Button>
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              {filteredModels.map((model) => {
                const alreadyExists = existingModelCodes.has(model.id);
                const canSelect = !alreadyExists && !adding;
                const isSelected = selectedIds.has(model.id);
                const modelDisplay = getModelDisplayParts({
                  name: model.displayName,
                  code: model.id,
                });

                return (
                  <button
                    key={model.id}
                    type="button"
                    onClick={() => toggleSelect(model.id)}
                    disabled={!canSelect}
                    className={cn(
                      "flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left outline-none transition-colors duration-150 motion-reduce:transition-none focus-visible:ring-2 focus-visible:ring-ring",
                      alreadyExists
                        ? "cursor-not-allowed border-transparent text-muted-foreground opacity-65"
                        : isSelected
                          ? "border-primary/35 bg-primary/8"
                          : "border-transparent hover:bg-muted/45",
                    )}
                  >
                    <div
                      className={cn(
                        "flex size-4 shrink-0 items-center justify-center rounded border transition-colors",
                        alreadyExists
                          ? "border-input bg-muted"
                          : isSelected
                            ? "border-primary bg-primary"
                            : "border-input",
                      )}
                    >
                      {(isSelected || alreadyExists) && (
                        <Check
                          className={cn(
                            "size-3",
                            alreadyExists ? "text-muted-foreground" : "text-primary-foreground",
                          )}
                        />
                      )}
                    </div>

                    <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-muted/60">
                      <ModelVendorIcon
                        source={{
                          name: model.displayName,
                          code: model.id,
                          platform: model.providerPlatform || apiConfig.platform,
                          capabilityPresetCode: model.capabilityPresetCode,
                        }}
                        className="size-4"
                      />
                    </span>

                    <div className="min-w-0 flex-1 space-y-0.5">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <p className="truncate text-sm font-medium text-foreground">
                          {modelDisplay.name}
                        </p>
                        {model.modelType != null && (
                          <Badge variant="secondary" className="text-xs">
                            {MODEL_TYPE_LABELS[model.modelType] || `类型${model.modelType}`}
                          </Badge>
                        )}
                        {model.capabilityPresetCode && (
                          <span
                            className="rounded-md bg-amber-500/12 px-1.5 py-0.5 text-xs text-amber-700 dark:text-amber-400"
                            title="已匹配内置模型能力预设"
                          >
                            预设：{model.capabilityPresetCode}
                          </span>
                        )}
                        {model.modelProtocol && model.modelProtocol !== "generic" && (
                          <span className="rounded-md bg-violet-500/10 px-1.5 py-0.5 text-xs text-violet-700 dark:text-violet-400">
                            {formatSemanticLabel(model.modelProtocol, MODEL_PROTOCOL_LABELS)}
                          </span>
                        )}
                      </div>
                      <div className="flex flex-wrap items-center gap-x-2 text-xs text-muted-foreground">
                        {modelDisplay.code && (
                          <span className="truncate font-mono">{modelDisplay.code}</span>
                        )}
                        {model.ownedBy && <span>{model.ownedBy}</span>}
                      </div>
                    </div>

                    {alreadyExists && (
                      <Badge variant="outline" className="shrink-0 text-xs">
                        已添加
                      </Badge>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </DialogScrollBody>

        <DialogFooter className="shrink-0">
          <DialogClose render={<Button variant="outline" disabled={adding} />}>
            取消
          </DialogClose>
          <Button
            type="button"
            onClick={handleAdd}
            disabled={adding || selectedIds.size === 0}
          >
            {adding && <Loader2 className="animate-spin" />}
            {addProgress
              ? `正在添加 ${addProgress.current}/${addProgress.total}…`
              : `添加${selectedIds.size > 0 ? ` (${selectedIds.size})` : ""}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
