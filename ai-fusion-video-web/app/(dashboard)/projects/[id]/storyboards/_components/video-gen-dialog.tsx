"use client";

import { useMemo, useState } from "react";
import { Video, X, ImageIcon, Check, Film, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { resolveMediaUrl } from "@/lib/api/client";
import type { StoryboardItem } from "@/lib/api/storyboard";

interface VideoGenDialogProps {
  open: boolean;
  onClose: () => void;
  /** 分镜条目列表 */
  items: StoryboardItem[];
  onConfirm: (selectedItemIds: number[], promptOnly?: boolean) => void;
}

export function VideoGenDialog({
  open,
  onClose,
  items,
  onConfirm,
}: VideoGenDialogProps) {
  const defaultSelected = useMemo(
    () => new Set(items.map((item) => item.id)),
    [items]
  );
  const [selectedOverride, setSelectedOverride] = useState<Set<number> | null>(null);
  const selected = selectedOverride ?? defaultSelected;

  if (!open) return null;

  const toggleItem = (id: number) => {
    setSelectedOverride((prev) => {
      const next = new Set(prev ?? selected);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === items.length) {
      setSelectedOverride(new Set());
    } else {
      setSelectedOverride(new Set(items.map((item) => item.id)));
    }
  };

  const handleConfirm = (promptOnly?: boolean) => {
    onConfirm(Array.from(selected), promptOnly);
    onClose();
  };

  /** 获取镜头的画面 URL（优先使用生成的） */
  const getImageUrl = (item: StoryboardItem) => {
    return item.firstFrameImageUrl || item.generatedImageUrl || item.imageUrl || null;
  };

  /** 景别标签 */
  const shotTypeLabels: Record<string, string> = {
    远景: "远景",
    全景: "全景",
    中景: "中景",
    近景: "近景",
    特写: "特写",
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* 遮罩 */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* 弹窗 */}
      <div className="relative bg-card border border-border/30 rounded-2xl shadow-2xl w-[520px] max-w-[90vw] max-h-[80vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border/20">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-linear-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center">
              <Video className="h-4 w-4 text-purple-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">批量生成视频</h3>
              <p className="text-[10px] text-muted-foreground">
                选择需要生成视频的分镜镜头
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* 全选 */}
        {items.length > 0 && (
          <div className="px-5 py-2.5 border-b border-border/10 flex items-center justify-between">
            <button
              onClick={toggleAll}
              className="text-xs text-primary hover:text-primary/80 font-medium transition-colors"
            >
              {selected.size === items.length ? "取消全选" : "全选"}
            </button>
            <span className="text-[10px] text-muted-foreground">
              已选 {selected.size} / {items.length} 个镜头
            </span>
          </div>
        )}

        {/* 镜头列表 */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1.5 min-h-0">
          {items.length === 0 ? (
            <div className="text-center py-8">
              <Film className="h-8 w-8 text-muted-foreground/20 mx-auto mb-2" />
              <p className="text-xs text-muted-foreground">暂无分镜镜头</p>
            </div>
          ) : (
            items.map((item) => {
              const checked = selected.has(item.id);
              const hasVideo = !!(item.videoUrl || item.generatedVideoUrl);
              const hasImage = !!(item.firstFrameImageUrl || item.imageUrl || item.generatedImageUrl);
              const imgUrl = getImageUrl(item);

              return (
                <button
                  key={item.id}
                  onClick={() => toggleItem(item.id)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all text-left",
                    checked
                      ? "bg-primary/8 ring-1 ring-primary/20"
                      : "hover:bg-muted/30"
                  )}
                >
                  {/* 选择指示器 */}
                  <div
                    className={cn(
                      "h-5 w-5 rounded-md border-2 flex items-center justify-center shrink-0 transition-all",
                      checked
                        ? "bg-primary border-primary text-primary-foreground"
                        : "border-border/50"
                    )}
                  >
                    {checked && <Check className="h-3 w-3" />}
                  </div>

                  {/* 缩略图 */}
                  <div className="h-12 w-20 rounded-lg bg-muted/30 border border-border/10 overflow-hidden shrink-0 flex items-center justify-center">
                    {imgUrl ? (
                      <img
                        src={resolveMediaUrl(imgUrl) || ""}
                        alt={`镜头 ${item.shotNumber || item.autoShotNumber}`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <ImageIcon className="h-4 w-4 text-muted-foreground/30" />
                    )}
                  </div>

                  {/* 信息 */}
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
                      {hasImage && (
                        <span className="text-[10px] text-cyan-400/70">
                          有首帧
                        </span>
                      )}
                      {hasVideo && (
                        <span className="text-[10px] text-green-400/70">
                          已有视频
                        </span>
                      )}
                    </div>
                  </div>
                </button>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3.5 border-t border-border/20 flex items-center justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-medium text-muted-foreground hover:bg-muted transition-colors"
          >
            取消
          </button>
          <button
            onClick={() => handleConfirm(true)}
            disabled={selected.size === 0}
            className={cn(
              "flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-medium transition-all",
              "border border-purple-500/30 bg-purple-500/10 text-purple-600 dark:text-purple-400",
              "hover:bg-purple-500/20 hover:shadow-sm",
              "active:scale-[0.98]",
              "disabled:opacity-40 disabled:pointer-events-none"
            )}
          >
            <FileText className="h-3.5 w-3.5" />
            只生提示词 ({selected.size})
          </button>
          <button
            onClick={() => handleConfirm(false)}
            disabled={selected.size === 0}
            className={cn(
              "flex items-center gap-1.5 px-5 py-2 rounded-xl text-xs font-medium transition-all",
              "bg-linear-to-r from-purple-600 to-pink-600 text-white",
              "hover:shadow-lg hover:shadow-purple-500/20 hover:scale-[1.02]",
              "active:scale-[0.98]",
              "disabled:opacity-40 disabled:pointer-events-none"
            )}
          >
            <Video className="h-3.5 w-3.5" />
            生成视频 ({selected.size})
          </button>
        </div>
      </div>
    </div>
  );
}
