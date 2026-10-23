"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { SafeImage } from "@/components/media/managed-image";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { assetApi, type AssetWithItems } from "@/lib/api/asset";
import { resolveMediaUrl } from "@/lib/api/client";
import { ChevronDown, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { typeLabelMap } from "./shared";

interface CoverSelectorProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: number | null;
  currentAssetId: number;
  onSelect: (url: string) => void;
}

export function CoverSelectorDialog(props: CoverSelectorProps) {
  return (
    <Dialog open={props.open} onOpenChange={props.onOpenChange}>
      {props.open && (
        <CoverSelectorBody
          key={String(props.projectId) + ":" + props.currentAssetId}
          {...props}
        />
      )}
    </Dialog>
  );
}

function CoverSelectorBody({
  onOpenChange,
  projectId,
  currentAssetId,
  onSelect,
}: CoverSelectorProps) {
  const [assets, setAssets] = useState<AssetWithItems[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());
  useEffect(() => {
    let active = true;
    assetApi
      .listWithItems(projectId)
      .then((result) => {
        if (!active) return;
        setAssets(
          [...result].sort(
            (a, b) =>
              Number(b.id === currentAssetId) - Number(a.id === currentAssetId),
          ),
        );
        setExpandedIds(new Set(result.map((asset) => asset.id)));
      })
      .catch((failure) => {
        if (active) setError(getApiErrorMessage(failure));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [projectId, currentAssetId, attempt]);

  return (
    <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
      <DialogHeader className="shrink-0">
        <DialogTitle>选择封面</DialogTitle>
        <DialogDescription>从可访问的素材中选择图片。</DialogDescription>
      </DialogHeader>
      <div className="min-h-0 min-w-0 space-y-3 overflow-y-auto scroll-p-1 p-1">
        {loading ? (
          <div className="flex items-center gap-2 py-6" role="status">
            <Loader2 className="size-4 animate-spin" />
            正在加载
          </div>
        ) : error ? (
          <div role="alert" className="space-y-2 text-destructive">
            <p>{error}</p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setLoading(true);
                setError("");
                setAttempt((value) => value + 1);
              }}
            >
              重试
            </Button>
          </div>
        ) : assets.length === 0 ? (
          <p className="py-6 text-sm text-muted-foreground">暂无素材</p>
        ) : (
          assets.map((asset) => (
            <section
              key={asset.id}
              className="space-y-3 border-b border-border/20 pb-4"
            >
              <button
                type="button"
                aria-expanded={expandedIds.has(asset.id)}
                className="flex min-h-9 w-full items-center gap-2 rounded-xl px-3 text-left text-sm outline-none hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring"
                onClick={() =>
                  setExpandedIds((current) => {
                    const next = new Set(current);
                    if (next.has(asset.id)) next.delete(asset.id);
                    else next.add(asset.id);
                    return next;
                  })
                }
              >
                <span className="flex-1 font-medium">{asset.name}</span>
                <span className="text-muted-foreground">
                  {typeLabelMap[asset.type] || asset.type}
                </span>
                <ChevronDown className="size-4" />
              </button>
              {expandedIds.has(asset.id) && (
                <div className="grid grid-cols-3 gap-3 sm:grid-cols-5">
                  {asset.items.length === 0 ? (
                    <p className="col-span-full text-sm text-muted-foreground">
                      暂无子资产
                    </p>
                  ) : (
                    asset.items.map((item) => {
                      const url = item.imageUrl || item.thumbnailUrl;
                      return (
                        <button
                          key={item.id}
                          type="button"
                          disabled={!url}
                          title={item.name || "未命名子资产"}
                          aria-label={"选用 " + (item.name || "未命名子资产")}
                          className="overflow-hidden rounded-xl border border-border/30 text-left outline-none hover:border-border/50 focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                          onClick={() => {
                            if (url) {
                              onSelect(url);
                              onOpenChange(false);
                            }
                          }}
                        >
                          <SafeImage
                            src={
                              resolveMediaUrl(
                                item.thumbnailUrl || item.imageUrl,
                              ) || undefined
                            }
                            alt={item.name || "子资产图片"}
                            className="aspect-square w-full object-cover"
                          />
                          <span className="block truncate p-2 text-xs">
                            {item.name || "未命名"}
                          </span>
                        </button>
                      );
                    })
                  )}
                </div>
              )}
            </section>
          ))
        )}
      </div>
    </DialogContent>
  );
}
