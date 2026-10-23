"use client";

import { SafeImage } from "@/components/media/managed-image";
import { resolveMediaUrl } from "@/lib/api/client";
import { cn } from "@/lib/utils";
import { ExternalLink, ZoomIn } from "lucide-react";
import { AssetItemWithParent, typeConfig } from "./reference-shared";

export /** 子资产分组展示 */
function AssetItemGroup({
  type,
  items,
  onItemClick,
  onPreviewImage,
}: {
  type: keyof typeof typeConfig;
  items: AssetItemWithParent[];
  onItemClick: (item: AssetItemWithParent) => void;
  onPreviewImage?: (url: string, title: string) => void;
}) {
  const config = typeConfig[type];
  const Icon = config.icon;

  return (
    <div className="border-t border-border/20 pt-3">
      <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
        <Icon className={cn("h-3 w-3", config.color)} />
        {config.label}
        <span className="text-[10px] font-normal text-muted-foreground/60 ml-auto">
          {items.length}
        </span>
      </h4>
      <div className="space-y-1.5">
        {items.map((item) => (
          <button
            key={item.id}
            onClick={() => onItemClick(item)}
            className={cn(
              "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg transition-all text-left group",
              "hover:bg-muted/30",
            )}
          >
            {/* 缩略图 */}
            <div
              onClick={(e) => {
                if (item.imageUrl && onPreviewImage) {
                  e.stopPropagation();
                  onPreviewImage(
                    item.imageUrl,
                    `${item.parentName}: ${item.name || "初始设定"}`,
                  );
                }
              }}
              className={cn(
                "h-10 w-10 bg-muted/30 border border-border/10 overflow-hidden shrink-0 flex items-center justify-center relative group/img",
                type === "character" ? "rounded-full" : "rounded-lg",
                item.imageUrl &&
                  "cursor-zoom-in hover:border-primary/40 transition-colors",
              )}
            >
              {item.imageUrl ? (
                <>
                  <SafeImage
                    src={resolveMediaUrl(item.imageUrl)}
                    alt={item.name || item.parentName}
                    fallbackType={
                      type === "character"
                        ? "avatar"
                        : type === "scene"
                          ? "scene"
                          : "prop"
                    }
                    className="w-full h-full object-cover transition-transform group-hover/img:scale-105"
                  />
                  <div className="absolute inset-0 bg-black/0 group-hover/img:bg-black/25 flex items-center justify-center opacity-0 group-hover/img:opacity-100 transition-all">
                    <ZoomIn className="h-3.5 w-3.5 text-white/90" />
                  </div>
                </>
              ) : (
                <Icon className={cn("h-4 w-4 text-muted-foreground/30")} />
              )}
            </div>
            {/* 信息 */}
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium truncate">
                {item.name || item.parentName}
              </p>
              <p className="text-[10px] text-muted-foreground/60 truncate mt-0.5">
                {item.parentName}
              </p>
            </div>
            {/* 跳转图标 */}
            <ExternalLink className="h-3 w-3 text-muted-foreground/30 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
          </button>
        ))}
      </div>
    </div>
  );
}
