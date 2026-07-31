"use client";

import { useRef } from "react";
import { ImageIcon, Images, Plus } from "lucide-react";
import type { AssetItem } from "@/lib/api/asset";
import { resolveMediaUrl } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/ui/safe-image";
import { cn } from "@/lib/utils";
import { getAppearance, getAssetType } from "./asset-types";

interface AssetItemTabsProps {
  items: AssetItem[];
  assetType: string;
  selectedItemId: number | null;
  creating: boolean;
  onSelect: (id: number) => void;
  onCreate: () => void;
  onGenerate: () => void;
}

export function AssetItemTabs({
  items,
  assetType,
  selectedItemId,
  creating,
  onSelect,
  onCreate,
  onGenerate,
}: AssetItemTabsProps) {
  const itemRefs = useRef<Record<number, HTMLButtonElement | null>>({});
  const type = getAssetType(assetType);
  const TypeIcon = type.icon;

  const select = (id: number) => {
    onSelect(id);
    itemRefs.current[id]?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
      inline: "center",
    });
  };

  return (
    <section className="min-w-0 border-b border-border/40 bg-card/50 px-5 pb-3.5 pt-3">
      <div className="mb-2.5 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold">子资产</h3>
          <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
            {items.length}
          </span>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={onGenerate}
          disabled={!items.length}
          title="批量生成当前主资产下的全部子资产图片"
          className="shrink-0"
        >
          <Images className="h-4 w-4" />
          批量子资产生图
        </Button>
      </div>

      <div className="flex snap-x snap-mandatory gap-2 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {items.map((item) => {
          const selected = !creating && selectedItemId === item.id;
          const appearance = getAppearance(item);

          return (
            <button
              key={item.id}
              ref={(node) => {
                itemRefs.current[item.id] = node;
              }}
              type="button"
              onClick={() => select(item.id)}
              className={cn(
                "flex h-[66px] min-w-[190px] snap-center items-center gap-2.5 rounded-2xl border p-2 text-left transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
                selected
                  ? "border-primary/35 bg-primary/[0.075] shadow-sm"
                  : "border-border/35 bg-background/45 hover:border-primary/25 hover:bg-background/80",
              )}
            >
              <div
                className={cn(
                  "grid h-11 w-11 shrink-0 place-items-center overflow-hidden rounded-xl",
                  type.bg,
                )}
              >
                {item.imageUrl ? (
                  <SafeImage
                    src={resolveMediaUrl(item.imageUrl) || undefined}
                    alt={item.name || "子资产"}
                    className="h-full w-full object-cover"
                    fallbackType="image"
                  />
                ) : (
                  <TypeIcon className={cn("h-5 w-5", type.color)} />
                )}
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-xs font-semibold">
                  {item.name || "未命名"}
                </p>
                <div className="mt-1 flex items-center gap-2 text-[10px] text-muted-foreground">
                  <span className="inline-flex items-center gap-1">
                    <span
                      className={cn(
                        "h-1.5 w-1.5 rounded-full",
                        item.imageUrl ? "bg-emerald-500" : "bg-amber-400",
                      )}
                    />
                    {item.imageUrl ? "有图片" : "待补图"}
                  </span>
                  <span className="truncate">
                    {appearance ? "已写外观" : "待写外观"}
                  </span>
                </div>
              </div>
            </button>
          );
        })}

        <button
          type="button"
          onClick={onCreate}
          className={cn(
            "flex h-[66px] min-w-[150px] snap-center items-center justify-center gap-2 rounded-2xl border border-dashed px-3 text-xs font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
            creating
              ? "border-primary/40 bg-primary/[0.075] text-primary shadow-sm"
              : "border-border/50 text-muted-foreground hover:border-primary/35 hover:bg-primary/5 hover:text-primary",
          )}
        >
          <span
            className={cn(
              "grid h-8 w-8 place-items-center rounded-xl",
              creating ? "bg-primary/10" : "bg-muted",
            )}
          >
            {creating ? (
              <ImageIcon className="h-4 w-4" />
            ) : (
              <Plus className="h-4 w-4" />
            )}
          </span>
          {creating ? "正在新建" : "新增子资产"}
        </button>
      </div>
    </section>
  );
}
