"use client";

import { Images, RefreshCw, Search } from "lucide-react";
import type { AssetWithItems } from "@/lib/api/asset";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SafeImage } from "@/components/ui/safe-image";
import { resolveMediaUrl } from "@/lib/api/client";
import { ASSET_TYPES, getAssetType } from "./asset-types";

interface AssetSidebarProps {
  assets: AssetWithItems[];
  selectedAssetId: number | null;
  search: string;
  typeFilter: string;
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
  onSearchChange,
  onTypeFilterChange,
  onSelect,
  onBatchGenerate,
  onRefresh,
}: AssetSidebarProps) {
  const filteredAssets = assets.filter((asset) => {
    const query = search.trim().toLowerCase();
    const matchesType = typeFilter === "all" || asset.type === typeFilter;
    const matchesSearch = !query || asset.name.toLowerCase().includes(query);
    return matchesType && matchesSearch;
  });

  return (
    <aside className="flex min-h-0 flex-col overflow-hidden rounded-[28px] border border-border/50 bg-card/80 shadow-[0_18px_50px_-32px_rgba(15,23,42,.35)]">
      <div className="border-b border-border/40 px-4 pb-3.5 pt-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold">资产目录</p>
            <p className="mt-0.5 text-[11px] text-muted-foreground">
              {filteredAssets.length} 个资产
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="image"
              size="sm"
              onClick={onBatchGenerate}
              disabled={!assets.length}
              title="批量生成资产目录中的全部子资产图片"
            >
              <Images className="h-3.5 w-3.5" />
              批量生图
            </Button>
            <button
              type="button"
              onClick={onRefresh}
              title="刷新"
              aria-label="刷新资产"
              className="grid h-8 w-8 place-items-center rounded-xl text-muted-foreground transition-colors hover:bg-primary/8 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>
        </div>
        <div className="relative mt-3">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground/50" />
          <Input
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder="搜索资产"
            className="h-9 rounded-xl bg-background/70 pl-9 text-xs"
          />
        </div>
        <div className="mt-3 flex gap-1.5 overflow-x-auto pb-0.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {[{ value: "all", label: "全部" }, ...ASSET_TYPES].map((item) => (
            <button
              key={item.value}
              type="button"
              onClick={() => onTypeFilterChange(item.value)}
              className={cn(
                "h-6 shrink-0 rounded-md border px-2 text-[10px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
                typeFilter === item.value
                  ? "border-primary/20 bg-primary/10 text-primary"
                  : "border-transparent text-muted-foreground hover:border-border/50 hover:bg-background/70",
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto p-3">
        {filteredAssets.length === 0 ? (
          <div className="flex h-48 flex-col items-center justify-center text-center text-xs text-muted-foreground">
            <Images className="mb-2 h-7 w-7 opacity-25" />
            暂无资产
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
                onClick={() => onSelect(asset)}
                className={cn(
                  "group relative flex w-full items-center gap-3 overflow-hidden rounded-2xl border p-2.5 text-left transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
                  selected
                    ? "border-primary/35 bg-primary/[0.07] shadow-sm"
                    : "border-transparent hover:border-border/50 hover:bg-background/70",
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
                  <p className="truncate text-xs font-semibold">{asset.name}</p>
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
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}
