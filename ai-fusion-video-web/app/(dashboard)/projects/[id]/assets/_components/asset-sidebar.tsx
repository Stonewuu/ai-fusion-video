"use client";

import { ChevronRight, Images, RefreshCw, Search } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { AssetWithItems } from "@/lib/api/asset";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { SafeImage } from "@/components/media/managed-image";
import { resolveMediaUrl } from "@/lib/api/client";
import { ASSET_TYPES, getAssetType } from "./asset-types";

export interface AssetSidebarProps {
  assets: AssetWithItems[];
  selectedAssetId: number | null;
  search: string;
  typeFilter: string;
  disabled: boolean;
  onSearchChange: (value: string) => void;
  onTypeFilterChange: (value: string) => void;
  onSelect: (asset: AssetWithItems) => void;
  onBatchGenerate: () => void;
  onRefresh: () => void;
}

export function AssetSidebar({
  assets,
  selectedAssetId,
  search,
  typeFilter,
  disabled,
  onSearchChange,
  onTypeFilterChange,
  onSelect,
  onBatchGenerate,
  onRefresh,
}: AssetSidebarProps) {
  const reduceMotion = useReducedMotion();
  const filteredAssets = assets.filter((asset) => {
    const query = search.trim().toLowerCase();
    const matchesType = typeFilter === "all" || asset.type === typeFilter;
    const matchesSearch = !query || asset.name.toLowerCase().includes(query);
    return matchesType && matchesSearch;
  });

  return (
    <aside
      data-sidebar-navigation="assets"
      className="flex h-full min-h-0 min-w-0 w-full flex-col overflow-hidden rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm"
    >
      <div className="shrink-0 border-b border-border/20 p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold">资产目录</p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {filteredAssets.length} 个资产
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="image"
              size="default"
              onClick={onBatchGenerate}
              disabled={disabled || !assets.length}
              title="批量生成资产目录中的全部子资产图片"
            >
              <Images />
              批量生图
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              disabled={disabled}
              onClick={onRefresh}
              title="刷新"
              aria-label="刷新资产"
            >
              <RefreshCw />
            </Button>
          </div>
        </div>
        <div className="relative mt-3">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground/50" />
          <Input
            disabled={disabled}
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder="搜索资产"
            aria-label="搜索资产"
            className="h-9 rounded-xl bg-background/70 pl-9 text-xs"
          />
        </div>
        <div className="mt-3 flex gap-2 overflow-x-auto p-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {[{ value: "all", label: "全部" }, ...ASSET_TYPES].map((item) => (
            <button
              key={item.value}
              type="button"
              disabled={disabled}
              onClick={() => onTypeFilterChange(item.value)}
              aria-pressed={typeFilter === item.value}
              className={cn(
                "h-6 shrink-0 rounded-lg border px-2 text-xs font-medium transition-colors duration-150 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 motion-reduce:transition-none",
                typeFilter === item.value
                  ? "border-primary/30 bg-primary/10 text-foreground"
                  : "border-transparent text-muted-foreground hover:border-border/30 hover:bg-muted/50",
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <OverlayScrollArea
        className="min-h-0 flex-1"
        viewportClassName="scroll-p-3"
      >
        <div className="p-3">
          <AnimatePresence initial={false} mode="wait">
            <motion.div
              key={filteredAssets.length === 0 ? "empty" : "list"}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{
                duration: reduceMotion ? 0 : 0.18,
                ease: "easeOut",
              }}
              className="flex flex-col gap-2"
            >
              {filteredAssets.length === 0 ? (
                <div className="flex h-48 flex-col items-center justify-center text-center text-xs text-muted-foreground">
                  <Images className="mb-2 h-7 w-7 opacity-25" />
                  {assets.length === 0 ? "暂无资产" : "没有匹配的资产"}
                </div>
              ) : (
                filteredAssets.map((asset) => {
                  const type = getAssetType(asset.type);
                  const TypeIcon = type.icon;
                  const selected = selectedAssetId === asset.id;

                  return (
                    <button
                      key={asset.id}
                      type="button"
                      disabled={disabled}
                      onClick={() => onSelect(asset)}
                      aria-current={selected ? "true" : undefined}
                      className={cn(
                        "group relative flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-colors duration-150 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 motion-reduce:transition-none",
                        selected
                          ? "border-primary/30 bg-primary/10"
                          : "border-transparent hover:border-border/30 hover:bg-muted/50",
                      )}
                    >
                      <div
                        className={cn(
                          "grid h-11 w-11 shrink-0 place-items-center overflow-hidden rounded-xl",
                          type.bg,
                        )}
                      >
                        {asset.coverUrl ? (
                          <SafeImage
                            src={resolveMediaUrl(asset.coverUrl) || undefined}
                            alt={asset.name}
                            className="h-full w-full object-cover"
                            fallbackType="image"
                          />
                        ) : (
                          <TypeIcon className={cn("h-5 w-5", type.color)} />
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-semibold lg:text-xs">
                          {asset.name}
                        </p>
                        <div className="mt-1 flex items-center gap-1.5">
                          <span
                            className={cn(
                              "rounded-md px-1.5 py-0.5 text-[9px] font-medium",
                              type.bg,
                              type.color,
                            )}
                          >
                            {type.label}
                          </span>
                          <span className="text-[10px] text-muted-foreground">
                            {asset.items?.length || 0} 个子资产
                          </span>
                        </div>
                      </div>
                      <ChevronRight className="size-4 shrink-0 text-muted-foreground lg:hidden" />
                    </button>
                  );
                })
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </OverlayScrollArea>
    </aside>
  );
}
