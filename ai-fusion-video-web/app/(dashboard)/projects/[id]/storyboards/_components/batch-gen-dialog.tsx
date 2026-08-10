"use client";

import { useMemo, useState } from "react";
import { Sparkles, X, ImageIcon, Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { resolveMediaUrl } from "@/lib/api/client";
import type { AssetItem } from "@/lib/api/asset";
import { SafeImage } from "@/components/ui/safe-image";

/** 带主资产信息的子资产 */
export interface AssetItemWithInfo {
  item: AssetItem;
  parentName: string;
  parentType: string;
  assetId: number;
}

/** 选中的子资产信息 */
export interface SelectedAssetItem {
  assetId: number;
  itemId: number;
}

interface BatchGenDialogProps {
  open: boolean;
  onClose: () => void;
  /** 子资产列表（带主资产信息） */
  assetItems: AssetItemWithInfo[];
  onConfirm: (selected: SelectedAssetItem[]) => void;
}

const typeLabels: Record<string, string> = {
  character: "角色",
  scene: "场景",
  prop: "道具",
  costume: "服装",
  vehicle: "载具",
};

const typeColors: Record<string, { bg: string; text: string }> = {
  character: { bg: "bg-blue-500/10", text: "text-blue-400" },
  scene: { bg: "bg-green-500/10", text: "text-green-400" },
  prop: { bg: "bg-amber-500/10", text: "text-amber-400" },
  costume: { bg: "bg-purple-500/10", text: "text-purple-400" },
  vehicle: { bg: "bg-teal-500/10", text: "text-teal-400" },
};

export function BatchGenDialog({
  open,
  onClose,
  assetItems,
  onConfirm,
}: BatchGenDialogProps) {
  const defaultSelected = useMemo(
    () => new Set(assetItems.map((asset) => asset.item.id)),
    [assetItems]
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
    if (selected.size === assetItems.length) {
      setSelectedOverride(new Set());
    } else {
      setSelectedOverride(new Set(assetItems.map((a) => a.item.id)));
    }
  };

  const handleConfirm = () => {
    const result: SelectedAssetItem[] = assetItems
      .filter((a) => selected.has(a.item.id))
      .map((a) => ({ assetId: a.assetId, itemId: a.item.id }));
    onConfirm(result);
    onClose();
  };

  // 按主资产分组展示子资产
  const grouped = new Map<number, { name: string; type: string; items: AssetItemWithInfo[] }>();
  for (const ai of assetItems) {
    const existing = grouped.get(ai.assetId);
    if (existing) {
      existing.items.push(ai);
    } else {
      grouped.set(ai.assetId, { name: ai.parentName, type: ai.parentType, items: [ai] });
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* 遮罩 */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* 弹窗 */}
      <div className="relative bg-card border border-border/30 rounded-2xl shadow-2xl w-[480px] max-w-[90vw] max-h-[80vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border/20">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-linear-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center">
              <Sparkles className="h-4 w-4 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">批量生图</h3>
              <p className="text-[10px] text-muted-foreground">
                选择需要生成图片的子资产
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
        {assetItems.length > 0 && (
          <div className="px-5 py-2.5 border-b border-border/10 flex items-center justify-between">
            <button
              onClick={toggleAll}
              className="text-xs text-primary hover:text-primary/80 font-medium transition-colors"
            >
              {selected.size === assetItems.length ? "取消全选" : "全选"}
            </button>
            <span className="text-[10px] text-muted-foreground">
              已选 {selected.size} / {assetItems.length}
            </span>
          </div>
        )}

        {/* 子资产列表（按主资产分组） */}
        <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0">
          {assetItems.length === 0 ? (
            <div className="text-center py-8">
              <ImageIcon className="h-8 w-8 text-muted-foreground/20 mx-auto mb-2" />
              <p className="text-xs text-muted-foreground">暂无子资产</p>
            </div>
          ) : (
            Array.from(grouped.entries()).map(([assetId, group]) => {
              const colors = typeColors[group.type] || typeColors.prop;
              return (
                <div key={assetId} className="space-y-1">
                  {/* 主资产标题 */}
                  <div className="flex items-center gap-2 px-2 py-1">
                    <span
                      className={cn(
                        "text-[10px] px-1.5 py-0.5 rounded font-medium",
                        colors.bg,
                        colors.text
                      )}
                    >
                      {typeLabels[group.type] || group.type}
                    </span>
                    <span className="text-xs font-medium text-muted-foreground">
                      {group.name}
                    </span>
                    <span className="text-[10px] text-muted-foreground/50 ml-auto">
                      {group.items.length} 个子资产
                    </span>
                  </div>

                  {/* 该主资产下的子资产 */}
                  {group.items.map((ai) => {
                    const checked = selected.has(ai.item.id);
                    return (
                      <button
                        key={ai.item.id}
                        onClick={() => toggleItem(ai.item.id)}
                        className={cn(
                          "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all text-left ml-1",
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
                        <SafeImage
                          src={resolveMediaUrl(ai.item.imageUrl) || undefined}
                          alt={ai.item.name || ai.parentName}
                          className="h-10 w-10 rounded-lg object-cover"
                          fallbackType={
                            ai.parentType === "character"
                              ? "avatar"
                              : ai.parentType === "scene"
                              ? "scene"
                              : ai.parentType === "prop"
                              ? "prop"
                              : "image"
                          }
                        />

                        {/* 信息 */}
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium truncate">
                            {ai.item.name || ai.parentName}
                          </p>
                          <div className="flex items-center gap-1.5 mt-0.5">
                            {ai.item.itemType && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted/40 text-muted-foreground">
                                {ai.item.itemType === "initial"
                                  ? "初始图"
                                  : ai.item.itemType === "three_view"
                                    ? "三视图"
                                    : ai.item.itemType === "variant"
                                      ? "变体"
                                      : ai.item.itemType}
                              </span>
                            )}
                            {ai.item.imageUrl && (
                              <span className="text-[10px] text-green-400/70">
                                已有图片
                              </span>
                            )}
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
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
            onClick={handleConfirm}
            disabled={selected.size === 0}
            className={cn(
              "flex items-center gap-1.5 px-5 py-2 rounded-xl text-xs font-medium transition-all",
              "bg-linear-to-r from-cyan-600 to-blue-600 text-white",
              "hover:shadow-lg hover:shadow-cyan-500/20 hover:scale-[1.02]",
              "active:scale-[0.98]",
              "disabled:opacity-40 disabled:pointer-events-none"
            )}
          >
            <Sparkles className="h-3.5 w-3.5" />
            开始生成 ({selected.size})
          </button>
        </div>
      </div>
    </div>
  );
}
