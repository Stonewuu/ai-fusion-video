"use client";

import { SafeImage } from "@/components/media/managed-image";
import { resolveMediaUrl } from "@/lib/api/client";
import type { StoryboardItem } from "@/lib/api/storyboard";
import { cn } from "@/lib/utils";
import { Check, Film, Image as ImageIcon, Sparkles, X } from "lucide-react";
import { useState } from "react";
import {
  BatchFrameGenerateHandler,
  compareStoryboardItemsAsc,
} from "./reference-shared";

export function BatchFrameGenDialog({
  open,
  items,
  episodeId,
  sceneId,
  onClose,
  onConfirm,
}: {
  open: boolean;
  items: StoryboardItem[];
  episodeId: number;
  sceneId: number;
  onClose: () => void;
  onConfirm: BatchFrameGenerateHandler;
}) {
  const [overwriteExisting, setOverwriteExisting] = useState(false);
  const [includeFirstFrame, setIncludeFirstFrame] = useState(true);
  const [includeLastFrame, setIncludeLastFrame] = useState(true);
  const [selectedOverride, setSelectedOverride] = useState<Set<number> | null>(
    null,
  );

  const hasFrameTypeSelected = includeFirstFrame || includeLastFrame;
  const sortedItems = [...items].sort(compareStoryboardItemsAsc);
  const selectableIds = new Set(sortedItems.map((item) => item.id));
  const defaultSelected = new Set(sortedItems.map((item) => item.id));
  const selected = selectedOverride
    ? new Set(
        Array.from(selectedOverride).filter((id) => selectableIds.has(id)),
      )
    : defaultSelected;
  const selectedItems = sortedItems.filter((item) => selected.has(item.id));
  const firstItemIds = includeFirstFrame
    ? selectedItems
        .filter((item) => overwriteExisting || !item.firstFrameImageUrl)
        .map((item) => item.id)
    : [];
  const lastItemIds = includeLastFrame
    ? selectedItems
        .filter((item) => overwriteExisting || !item.lastFrameImageUrl)
        .map((item) => item.id)
    : [];
  const generateCount = firstItemIds.length + lastItemIds.length;
  const canSubmit = hasFrameTypeSelected && generateCount > 0;

  if (!open) return null;

  const resetSelection = () => {
    setSelectedOverride(null);
  };

  const toggleItem = (id: number) => {
    setSelectedOverride((prev) => {
      const next = new Set(prev ?? selected);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === sortedItems.length) {
      setSelectedOverride(new Set());
    } else {
      setSelectedOverride(new Set(sortedItems.map((item) => item.id)));
    }
  };

  const getPreviewUrl = (item: StoryboardItem) =>
    item.firstFrameImageUrl ||
    item.generatedImageUrl ||
    item.imageUrl ||
    item.referenceImageUrl ||
    item.lastFrameImageUrl ||
    null;

  const shotTypeLabels: Record<string, string> = {
    远景: "远景",
    全景: "全景",
    中景: "中景",
    近景: "近景",
    特写: "特写",
  };

  const handleConfirm = async () => {
    if (!canSubmit) return;
    await onConfirm({
      episodeId,
      sceneId,
      firstItemIds,
      lastItemIds,
      overwriteExisting,
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      <div className="relative bg-card border border-border/30 rounded-xl shadow-2xl w-[520px] max-w-[92vw] max-h-[82vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-5 py-4 border-b border-border/20">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <Sparkles className="h-4 w-4 text-primary" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">批量生成首尾帧</h3>
              <p className="text-[10px] text-muted-foreground">
                选择当前场次中需要生成首尾帧的分镜镜头
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground transition-colors"
            title="关闭"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4 border-b border-border/10">
          <div className="rounded-lg border border-border/30 bg-muted/10 px-3 py-3">
            <p className="text-xs font-medium mb-2">生成类型</p>
            <div className="grid grid-cols-2 gap-2">
              <label className="flex items-center gap-2 rounded-lg bg-background/40 px-3 py-2 text-xs">
                <input
                  type="checkbox"
                  checked={includeFirstFrame}
                  onChange={(event) => {
                    setIncludeFirstFrame(event.target.checked);
                    resetSelection();
                  }}
                  className="h-4 w-4 rounded border-border accent-primary"
                />
                <span>首帧</span>
              </label>
              <label className="flex items-center gap-2 rounded-lg bg-background/40 px-3 py-2 text-xs">
                <input
                  type="checkbox"
                  checked={includeLastFrame}
                  onChange={(event) => {
                    setIncludeLastFrame(event.target.checked);
                    resetSelection();
                  }}
                  className="h-4 w-4 rounded border-border accent-primary"
                />
                <span>尾帧</span>
              </label>
            </div>
          </div>

          <label className="flex items-start gap-3 rounded-lg border border-border/30 bg-muted/10 px-3 py-3 text-left">
            <input
              type="checkbox"
              checked={overwriteExisting}
              onChange={(event) => {
                setOverwriteExisting(event.target.checked);
                resetSelection();
              }}
              className="mt-0.5 h-4 w-4 rounded border-border accent-primary"
            />
            <span className="min-w-0">
              <span className="block text-xs font-medium">覆盖已有首尾帧</span>
              <span className="mt-1 block text-[10px] leading-relaxed text-muted-foreground">
                默认关闭时只补齐所选镜头中缺失的首帧或尾帧；开启后会重新生成所选镜头的首帧和尾帧。
              </span>
            </span>
          </label>

          {!hasFrameTypeSelected ? (
            <div className="rounded-lg bg-amber-500/10 border border-amber-500/20 px-3 py-2.5 text-xs text-amber-600">
              请至少选择首帧或尾帧中的一种。
            </div>
          ) : canSubmit ? (
            <div className="rounded-lg bg-primary/5 border border-primary/15 px-3 py-2.5 text-xs text-primary">
              将提交 {firstItemIds.length} 个首帧和 {lastItemIds.length}{" "}
              个尾帧生成任务。
            </div>
          ) : (
            <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 px-3 py-2.5 text-xs text-emerald-600">
              当前场次首尾帧已完整，无需生成。
            </div>
          )}
        </div>

        {sortedItems.length > 0 && (
          <div className="px-5 py-2.5 border-b border-border/10 flex items-center justify-between">
            <button
              type="button"
              onClick={toggleAll}
              className="text-xs text-primary hover:text-primary/80 font-medium transition-colors"
            >
              {selected.size === sortedItems.length ? "取消全选" : "全选"}
            </button>
            <span className="text-[10px] text-muted-foreground">
              已选 {selected.size} / {sortedItems.length} 个镜头
            </span>
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-3 space-y-1.5 min-h-0">
          {sortedItems.length === 0 ? (
            <div className="text-center py-8">
              <Film className="h-8 w-8 text-muted-foreground/20 mx-auto mb-2" />
              <p className="text-xs text-muted-foreground">当前场次暂无镜头</p>
            </div>
          ) : (
            sortedItems.map((item) => {
              const checked = selected.has(item.id);
              const imgUrl = getPreviewUrl(item);
              const willGenerateFirst =
                checked &&
                includeFirstFrame &&
                (overwriteExisting || !item.firstFrameImageUrl);
              const willGenerateLast =
                checked &&
                includeLastFrame &&
                (overwriteExisting || !item.lastFrameImageUrl);

              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => toggleItem(item.id)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all text-left",
                    checked
                      ? "bg-primary/8 ring-1 ring-primary/20"
                      : "hover:bg-muted/30",
                  )}
                >
                  <div
                    className={cn(
                      "h-5 w-5 rounded-md border-2 flex items-center justify-center shrink-0 transition-all",
                      checked
                        ? "bg-primary border-primary text-primary-foreground"
                        : "border-border/50",
                    )}
                  >
                    {checked && <Check className="h-3 w-3" />}
                  </div>

                  <div className="h-12 w-20 rounded-lg bg-muted/30 border border-border/10 overflow-hidden shrink-0 flex items-center justify-center">
                    {imgUrl ? (
                      <SafeImage
                        src={resolveMediaUrl(imgUrl) || undefined}
                        alt={`镜头 ${item.shotNumber || item.autoShotNumber || ""}`}
                        fallbackType="image"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <ImageIcon className="h-4 w-4 text-muted-foreground/30" />
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <p className="text-xs font-medium">
                        #{item.shotNumber || item.autoShotNumber || "?"}
                      </p>
                      {item.shotType && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted/40 text-muted-foreground">
                          {shotTypeLabels[item.shotType] || item.shotType}
                        </span>
                      )}
                      {item.cameraMovement && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400">
                          {item.cameraMovement}
                        </span>
                      )}
                    </div>
                    <p className="text-[10px] text-muted-foreground truncate mt-0.5">
                      {item.content || "（无画面描述）"}
                    </p>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      {willGenerateFirst && (
                        <span className="text-[10px] text-cyan-400/70">
                          生成首帧
                        </span>
                      )}
                      {willGenerateLast && (
                        <span className="text-[10px] text-emerald-400/70">
                          生成尾帧
                        </span>
                      )}
                      {item.firstFrameImageUrl && (
                        <span className="text-[10px] text-muted-foreground/60">
                          已有首帧
                        </span>
                      )}
                      {item.lastFrameImageUrl && (
                        <span className="text-[10px] text-muted-foreground/60">
                          已有尾帧
                        </span>
                      )}
                    </div>
                  </div>
                </button>
              );
            })
          )}
        </div>

        <div className="px-5 py-3.5 border-t border-border/20 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs font-medium text-muted-foreground hover:bg-muted transition-colors"
          >
            取消
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            disabled={!canSubmit}
            className={cn(
              "flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium transition-all",
              "bg-primary text-primary-foreground hover:bg-primary/90",
              "disabled:opacity-40 disabled:pointer-events-none",
            )}
          >
            <Sparkles className="h-3.5 w-3.5" />
            开始生成 ({generateCount})
          </button>
        </div>
      </div>
    </div>
  );
}

export function BatchFrameGenerateControl({
  items,
  loading,
  currentEpisodeId,
  currentSceneId,
  onConfirm,
}: {
  items: StoryboardItem[];
  loading?: boolean;
  currentEpisodeId?: number | null;
  currentSceneId?: number | null;
  onConfirm?: BatchFrameGenerateHandler;
}) {
  const [open, setOpen] = useState(false);
  const disabledReason =
    currentSceneId == null
      ? "请先选择场次"
      : currentEpisodeId == null
        ? "缺少分镜集上下文"
        : loading
          ? "正在加载当前场次镜头"
          : items.length === 0
            ? "当前场次暂无镜头"
            : !onConfirm
              ? "当前页面暂不支持批量生成"
              : null;
  const disabled = !!disabledReason;

  return (
    <div className="space-y-1.5">
      <button
        type="button"
        onClick={() => {
          if (!disabled) {
            setOpen(true);
          }
        }}
        disabled={disabled}
        title={disabledReason || "批量生成当前场次全部镜头的首尾帧"}
        className={cn(
          "w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-medium transition-all",
          disabled
            ? "bg-muted/30 text-muted-foreground cursor-not-allowed"
            : "bg-linear-to-r from-emerald-600 to-teal-600 text-white hover:shadow-lg hover:shadow-emerald-500/20 hover:scale-[1.02] active:scale-[0.98]",
        )}
      >
        <ImageIcon className="h-3.5 w-3.5" />
        批量生成首尾帧
      </button>
      {disabledReason && (
        <p className="text-[10px] text-muted-foreground/70 text-center">
          {disabledReason}
        </p>
      )}
      {onConfirm && currentEpisodeId != null && currentSceneId != null && (
        <BatchFrameGenDialog
          key={open ? "batch-frame-open" : "batch-frame-closed"}
          open={open}
          items={items}
          episodeId={currentEpisodeId}
          sceneId={currentSceneId}
          onClose={() => setOpen(false)}
          onConfirm={onConfirm}
        />
      )}
    </div>
  );
}
