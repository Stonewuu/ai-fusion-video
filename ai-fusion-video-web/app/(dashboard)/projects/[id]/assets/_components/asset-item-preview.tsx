"use client";

import { Edit3, Images, Trash2 } from "lucide-react";
import type { AssetItem } from "@/lib/api/asset";
import { SafeImage } from "@/components/media/managed-image";
import { Button } from "@/components/ui/button";
import { resolveMediaUrl } from "@/lib/api/client";
import { getAppearance } from "./asset-types";

export function AssetItemPreview({
  item,
  disabled,
  onEdit,
  onEditImage,
  onDelete,
  onGenerate,
}: {
  item: AssetItem;
  disabled: boolean;
  onEdit: () => void;
  onEditImage: () => void;
  onDelete: () => void;
  onGenerate: () => void;
}) {
  const appearance = getAppearance(item);
  return (
    <section
      aria-label="子资产详情"
      className="flex min-w-0 flex-col gap-4 p-5"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground">子资产</p>
          <h3 className="mt-1 break-words text-lg font-semibold">
            {item.name || "未命名"}
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            disabled={disabled}
            onClick={onEdit}
            aria-label="编辑子资产"
          >
            <Edit3 />
            编辑
          </Button>
          <Button
            variant="destructive-ghost"
            size="icon"
            disabled={disabled}
            onClick={onDelete}
            aria-label="删除子资产"
          >
            <Trash2 />
          </Button>
        </div>
      </div>
      <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <div className="flex min-w-0 flex-col gap-3">
          <div className="flex aspect-[16/10] w-full items-center justify-center overflow-hidden rounded-lg border border-border/20 bg-background/70">
            {item.imageUrl ? (
              <SafeImage
                src={resolveMediaUrl(item.imageUrl) || undefined}
                alt={item.name || "子资产"}
                className="h-full w-full object-contain p-3"
                fallbackType="image"
              />
            ) : (
              <div className="flex flex-col items-center gap-2 text-muted-foreground">
                <Images className="size-7 opacity-50" />
                <p className="text-sm">暂无图片</p>
              </div>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={onEditImage} disabled={disabled}>
              编辑图片
            </Button>
            <Button variant="image" onClick={onGenerate} disabled={disabled}>
              <Images />
              生成图片
            </Button>
          </div>
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <h4 className="text-sm font-medium text-muted-foreground">
            外观描写
          </h4>
          <p className="whitespace-pre-wrap break-words text-sm leading-7">
            {appearance || "—"}
          </p>
        </div>
      </div>
    </section>
  );
}
