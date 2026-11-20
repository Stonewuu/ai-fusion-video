"use client";

import { useCallback, useEffect, useRef } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ImageIcon, Images, Plus } from "lucide-react";
import type { AssetItem } from "@/lib/api/asset";
import { resolveMediaUrl } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/media/managed-image";
import { cn } from "@/lib/utils";
import { getAppearance, getAssetType } from "./asset-types";

interface AssetItemTabsProps {
  items: AssetItem[];
  assetType: string;
  selectedItemId: number | null;
  creating: boolean;
  disabled: boolean;
  onSelect: (id: number) => void;
  onCreate: () => void;
  onGenerate: () => void;
}

export function AssetItemTabs({
  items,
  assetType,
  selectedItemId,
  creating,
  disabled,
  onSelect,
  onCreate,
  onGenerate,
}: AssetItemTabsProps) {
  const itemRefs = useRef<Record<number, HTMLButtonElement | null>>({});
  const scrollRef = useRef<HTMLDivElement>(null);
  const createRef = useRef<HTMLButtonElement>(null);
  const reduceMotion = useReducedMotion();
  const type = getAssetType(assetType);
  const TypeIcon = type.icon;

  const scrollToSelection = useCallback(() => {
    const viewport = scrollRef.current;
    const selected = creating
      ? createRef.current
      : selectedItemId === null
        ? null
        : itemRefs.current[selectedItemId];
    if (!viewport || !selected) {
      return;
    }
    viewport.scrollTo({
      left:
        selected.offsetLeft - (viewport.clientWidth - selected.offsetWidth) / 2,
      behavior: reduceMotion ? "auto" : "smooth",
    });
  }, [selectedItemId, creating, reduceMotion]);

  useEffect(scrollToSelection, [scrollToSelection, items.length]);

  return (
    <section className="min-w-0 border-b border-border/20 px-5 py-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold">子资产</h3>
          <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
            {items.length}
          </span>
        </div>
        <Button
          variant="secondary"
          size="default"
          onClick={onGenerate}
          disabled={disabled || !items.length}
          title="批量生成当前主资产下的全部子资产图片"
          className="shrink-0"
        >
          <Images className="h-4 w-4" />
          批量子资产生图
        </Button>
      </div>

      <div
        ref={scrollRef}
        className="relative flex snap-x snap-mandatory gap-2 overflow-x-auto p-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      >
        <AnimatePresence initial={false} onExitComplete={scrollToSelection}>
          {items.map((item) => {
            const selected = !creating && selectedItemId === item.id;
            const appearance = getAppearance(item);

            return (
              <motion.button
                key={item.id}
                ref={(node) => {
                  itemRefs.current[item.id] = node;
                }}
                type="button"
                disabled={disabled}
                onClick={() => onSelect(item.id)}
                aria-pressed={selected}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{
                  opacity: 0,
                  transition: {
                    duration: reduceMotion ? 0 : 0.15,
                    ease: "easeIn",
                  },
                }}
                transition={{
                  duration: reduceMotion ? 0 : 0.2,
                  ease: [0.25, 0.46, 0.45, 0.94],
                }}
                className={cn(
                  "flex h-16 min-w-48 snap-center items-center gap-3 rounded-lg border p-2 text-left transition-colors duration-150 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 motion-reduce:transition-none",
                  selected
                    ? "border-primary/30 bg-primary/10"
                    : "border-border/20 bg-background/70 hover:border-border/50 hover:bg-muted/50",
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
              </motion.button>
            );
          })}
        </AnimatePresence>
        <button
          ref={createRef}
          type="button"
          disabled={disabled}
          onClick={onCreate}
          aria-pressed={creating}
          className={cn(
            "flex h-16 min-w-40 snap-center items-center justify-center gap-2 rounded-lg border border-dashed px-3 text-xs font-medium transition-colors duration-150 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 motion-reduce:transition-none",
            creating
              ? "border-primary/30 bg-primary/10 text-foreground"
              : "border-border/30 text-muted-foreground hover:border-border/50 hover:bg-muted/50 hover:text-foreground",
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
