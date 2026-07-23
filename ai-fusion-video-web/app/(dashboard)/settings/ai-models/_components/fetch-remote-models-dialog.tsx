"use client";

import { useCallback, useEffect, useState } from "react";
import { Check, Loader2, Search } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  aiModelApi,
  apiConfigApi,
  MODEL_TYPE_OPTIONS,
  MODEL_TYPE_LABELS,
  type ApiConfig,
  type RemoteModel,
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
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
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
  const [searchQuery, setSearchQuery] = useState("");
  const [modelType, setModelType] = useState<number>(1);

  const hasUnknownModelTypes = remoteModels.some(model => model.modelType == null);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const models = await apiConfigApi.remoteModels(apiConfig.id);
      setRemoteModels(models);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "获取模型列表失败";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [apiConfig.id]);

  useEffect(() => {
    if (open) {
      setRemoteModels([]);
      setError(null);
      setSelectedIds(new Set());
      setSearchQuery("");
      fetchModels();
    }
  }, [fetchModels, open]);

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
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
    const filtered = filteredModels;
    const allSelected = filtered.every(m => selectedIds.has(m.id));
    if (allSelected) {
      const next = new Set(selectedIds);
      filtered.forEach(m => next.delete(m.id));
      setSelectedIds(next);
    } else {
      const next = new Set(selectedIds);
      filtered.forEach(m => {
        if (!existingModelCodes.has(m.id)) {
          next.add(m.id);
        }
      });
      setSelectedIds(next);
    }
  };

  const handleAdd = async () => {
    if (selectedIds.size === 0) return;
    setAdding(true);
    try {
      for (const modelId of selectedIds) {
        const remoteModel = remoteModels.find(model => model.id === modelId);
        await aiModelApi.create({
          name: remoteModel?.displayName || modelId,
          code: modelId,
          capabilityPresetCode: remoteModel?.capabilityPresetCode || undefined,
          modelProtocol: remoteModel?.modelProtocol || undefined,
          modelType: remoteModel?.modelType ?? modelType,
          apiConfigId: apiConfig.id,
        });
      }
      onAdded();
      onOpenChange(false);
    } catch (err) {
      console.error("添加模型失败:", err);
    } finally {
      setAdding(false);
    }
  };

  const filteredModels = remoteModels.filter(m => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return m.id.toLowerCase().includes(q) || (m.ownedBy && m.ownedBy.toLowerCase().includes(q));
  });

  const selectableCount = filteredModels.filter(m => !existingModelCodes.has(m.id)).length;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-4xl max-h-[calc(100vh-2rem)] sm:max-h-[88vh] flex flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>获取可用模型</DialogTitle>
          <DialogDescription>
            从 {apiConfig.name} 获取远程可用模型列表，选择后点击添加
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3 min-h-0 overflow-y-auto px-1 pt-1 -mx-1">
          {/* 搜索框 + 模型类型选择 */}
          <div className="flex items-center gap-2 shrink-0">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                placeholder="搜索模型..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="text-sm pl-8 h-8"
              />
            </div>
            {hasUnknownModelTypes ? (
              <Select
                value={modelType}
                onValueChange={v => setModelType(v as number)}
                items={MODEL_TYPE_OPTIONS.map(o => ({ value: o.value, label: o.label }))}
              >
                <SelectTrigger className="w-[148px] text-xs h-8">
                  <SelectValue placeholder="默认模型类型" />
                </SelectTrigger>
                <SelectContent className="text-xs">
                  <SelectGroup>
                    {MODEL_TYPE_OPTIONS.map(opt => (
                      <SelectItem key={opt.value} value={opt.value} className="text-xs">
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            ) : (
              <div className="inline-flex h-8 items-center rounded-lg border border-emerald-500/20 bg-emerald-500/8 px-2.5 text-[10px] text-emerald-600 shrink-0">
                已自动识别模型类型
              </div>
            )}
          </div>

          {hasUnknownModelTypes && (
            <p className="-mt-1 text-[10px] text-muted-foreground">
              已识别类型的模型会按返回值自动导入；仅未识别类型的模型才会使用右侧默认类型。
            </p>
          )}

          {/* 模型列表 */}
          {loading ? (
            <div className="flex items-center justify-center py-10">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              <span className="ml-2 text-sm text-muted-foreground">正在获取模型列表...</span>
            </div>
          ) : error ? (
            <div className="text-center py-8 space-y-2">
              <p className="text-sm text-destructive">{error}</p>
              <Button variant="outline" size="sm" onClick={fetchModels}>
                重试
              </Button>
            </div>
          ) : remoteModels.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-sm text-muted-foreground">未获取到模型</p>
            </div>
          ) : (
            <>
              {/* 全选 + 计数 */}
              <div className="flex items-center justify-between px-1 shrink-0">
                <button
                  type="button"
                  onClick={toggleSelectAll}
                  className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {filteredModels.filter(m => !existingModelCodes.has(m.id)).every(m => selectedIds.has(m.id)) && selectableCount > 0
                    ? "取消全选"
                    : "全选可添加"}
                </button>
                <span className="text-[10px] text-muted-foreground">
                  共 {filteredModels.length} 个模型，已选 {selectedIds.size} 个
                </span>
              </div>

              <div className="overflow-y-auto min-h-0 max-h-[400px] -mx-1 px-1 space-y-0.5">
                {filteredModels.map(model => {
                  const alreadyExists = existingModelCodes.has(model.id);
                  const isSelected = selectedIds.has(model.id);

                  return (
                    <button
                      key={model.id}
                      type="button"
                      onClick={() => !alreadyExists && toggleSelect(model.id)}
                      disabled={alreadyExists}
                      className={cn(
                        "flex items-center gap-3 w-full px-3 py-2 rounded-lg text-left transition-all duration-150",
                        alreadyExists
                          ? "opacity-50 cursor-not-allowed"
                          : isSelected
                            ? "bg-primary/10 border border-primary/30"
                            : "hover:bg-muted/50 border border-transparent"
                      )}
                    >
                      {/* 选择框 */}
                      <div className={cn(
                        "w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-colors",
                        alreadyExists
                          ? "bg-muted border-border"
                          : isSelected
                            ? "bg-primary border-primary"
                            : "border-border/60"
                      )}>
                        {(isSelected || alreadyExists) && (
                          <Check className={cn("h-3 w-3", alreadyExists ? "text-muted-foreground" : "text-primary-foreground")} />
                        )}
                      </div>

                      {/* 模型信息 */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <p className="text-sm font-mono truncate">{model.id}</p>
                          {model.modelType != null && (
                            <span className={cn(
                              "shrink-0 rounded px-1.5 py-0.5 text-[10px]",
                              model.modelType === 2
                                ? "bg-sky-500/10 text-sky-500"
                                : model.modelType === 3
                                  ? "bg-emerald-500/10 text-emerald-500"
                                  : "bg-muted text-muted-foreground"
                            )}>
                              {MODEL_TYPE_LABELS[model.modelType] || `类型${model.modelType}`}
                            </span>
                          )}
                          {model.capabilityPresetCode && (
                            <span className="shrink-0 rounded px-1.5 py-0.5 text-[10px] bg-amber-500/10 text-amber-600" title="已精确匹配内置模型能力预设">
                              能力预设：{model.capabilityPresetCode}
                            </span>
                          )}
                          {model.modelProtocol && model.modelProtocol !== "generic" && (
                            <span className="shrink-0 rounded px-1.5 py-0.5 text-[10px] bg-violet-500/10 text-violet-600">
                              {formatSemanticLabel(model.modelProtocol, MODEL_PROTOCOL_LABELS)}
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-muted-foreground space-y-0.5">
                          {model.displayName && model.displayName !== model.id && (
                            <p>{model.displayName}</p>
                          )}
                          {model.ownedBy && (
                            <p>{model.ownedBy}</p>
                          )}
                        </div>
                      </div>

                      {/* 已添加标记 */}
                      {alreadyExists && (
                        <span className="text-[10px] text-muted-foreground shrink-0 px-1.5 py-0.5 rounded bg-muted">
                          已添加
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </>
          )}
        </div>

        <DialogFooter className="shrink-0">
          <DialogClose render={<Button variant="outline" size="sm" />}>
            取消
          </DialogClose>
          <Button
            size="sm"
            onClick={handleAdd}
            disabled={adding || selectedIds.size === 0}
          >
            {adding && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
            添加 {selectedIds.size > 0 ? `(${selectedIds.size})` : ""}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
