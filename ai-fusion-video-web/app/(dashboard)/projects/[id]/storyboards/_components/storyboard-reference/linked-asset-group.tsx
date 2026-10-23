"use client";

import { SafeImage } from "@/components/media/managed-image";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Asset, AssetItem } from "@/lib/api/asset";
import { resolveMediaUrl } from "@/lib/api/client";
import { cn } from "@/lib/utils";
import { ZoomIn } from "lucide-react";
import { typeConfig } from "./reference-shared";

export /** 关联资产分组展示（和表格页一致的上下布局，左右排列，无边框） */
function LinkedAssetGroup({
  type,
  assets,
  onAssetClick,
  onPreviewImage,
}: {
  type: keyof typeof typeConfig;
  assets: (Asset & { items: AssetItem[] })[];
  onAssetClick: (id: number) => void;
  onPreviewImage?: (url: string, title: string) => void;
}) {
  const config = typeConfig[type];
  const Icon = config.icon;

  // 将主资产与其关联的子资产打平，展示为独立的项，和表格页完全一致
  const itemsToShow = assets.flatMap((asset) => {
    if (asset.items.length === 0) {
      return [{ asset, sub: null as AssetItem | null }];
    }
    return asset.items.map((sub) => ({ asset, sub: sub as AssetItem | null }));
  });

  return (
    <div className="border-t border-border/20 pt-3">
      <h5 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
        <Icon className={cn("h-3 w-3", config.color)} />
        {config.label}
        <span className="text-[10px] font-normal text-muted-foreground/60 ml-auto">
          {itemsToShow.length}
        </span>
      </h5>

      <div className="flex flex-row flex-wrap gap-x-4 gap-y-3.5 px-1 py-1">
        {itemsToShow.map(({ asset, sub }, index) => {
          const displayName =
            !sub ||
            !sub.name ||
            sub.name === "初始设定" ||
            sub.name === asset.name
              ? asset.name
              : sub.name;

          const imageUrl = sub?.imageUrl || asset.coverUrl;
          const letter = displayName.charAt(0);

          // 根据类型定制颜色
          const themeClass =
            type === "character"
              ? "bg-blue-500/10 text-blue-500 border-blue-500/20 text-blue-500 dark:text-blue-400 group-hover/asset:text-blue-600"
              : type === "scene"
                ? "bg-green-500/10 text-green-500 border-green-500/20 text-green-500 dark:text-green-400 group-hover/asset:text-green-600"
                : "bg-amber-500/10 text-amber-500 border-amber-500/20 text-amber-500 dark:text-amber-400 group-hover/asset:text-amber-600";

          return (
            <Tooltip key={`${asset.id}-${sub?.id || index}`}>
              <TooltipTrigger
                render={
                  <div
                    onClick={() => onAssetClick(asset.id)}
                    className="flex flex-col items-center group/asset cursor-pointer select-none shrink-0"
                  >
                    <div className="relative">
                      {imageUrl ? (
                        <div
                          onClick={(e) => {
                            if (onPreviewImage) {
                              e.stopPropagation();
                              onPreviewImage(
                                imageUrl,
                                `${config.label}: ${displayName}`,
                              );
                            }
                          }}
                          className={cn(
                            "h-11 w-11 bg-muted/30 border border-border/10 overflow-hidden shrink-0 flex items-center justify-center relative group/coverimg cursor-zoom-in",
                            type === "character"
                              ? "rounded-full"
                              : "rounded-lg",
                          )}
                        >
                          <SafeImage
                            src={resolveMediaUrl(imageUrl)}
                            alt={displayName}
                            fallbackType={
                              type === "character"
                                ? "avatar"
                                : type === "scene"
                                  ? "scene"
                                  : "prop"
                            }
                            className="w-full h-full object-cover transition-transform duration-200 group-hover/asset:scale-110"
                          />
                          <div className="absolute inset-0 bg-black/0 group-hover/coverimg:bg-black/25 flex items-center justify-center opacity-0 group-hover/coverimg:opacity-100 transition-all">
                            <ZoomIn className="h-3 w-3 text-white/90" />
                          </div>
                        </div>
                      ) : (
                        <div
                          className={cn(
                            "h-11 w-11 border flex items-center justify-center font-semibold text-xs shrink-0 transition-colors duration-200",
                            type === "character"
                              ? "rounded-full"
                              : "rounded-lg",
                            themeClass.split(" ").slice(0, 3).join(" "), // 只提取 bg, text, border 相关类
                          )}
                        >
                          {letter}
                        </div>
                      )}
                    </div>
                    <span
                      className={cn(
                        "text-[10px] font-semibold truncate max-w-[56px] leading-tight text-center mt-1.5 transition-colors duration-200",
                        themeClass.split(" ").slice(3).join(" "), // 提取文字颜色类
                      )}
                    >
                      {displayName}
                    </span>
                  </div>
                }
              />
              <TooltipContent className="max-w-xs flex flex-col gap-1 text-left items-start px-3.5 py-2.5 rounded-xl text-xs leading-normal bg-white/85 dark:bg-zinc-900/85 backdrop-blur-md border border-zinc-200/50 dark:border-zinc-800/50 text-zinc-900 dark:text-zinc-50 shadow-lg [&_.bg-foreground]:bg-white/85 [&_.fill-foreground]:fill-white/85 dark:[&_.bg-foreground]:bg-zinc-900/85 dark:[&_.fill-foreground]:fill-zinc-900/85">
                <span className="font-semibold text-xs">
                  {config.label}: {displayName}
                </span>
                {asset.description && (
                  <span className="text-[10px] opacity-80 leading-normal max-w-[200px] break-words mt-0.5">
                    {asset.description}
                  </span>
                )}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </div>
  );
}
