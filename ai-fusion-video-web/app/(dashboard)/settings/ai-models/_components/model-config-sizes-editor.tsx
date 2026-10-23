"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChevronRight, Trash2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { COMMON_ASPECT_RATIOS, COMMON_TIERS } from "./model-config-options";

export type SizesMap = Record<string, Record<string, string>>;

export function SupportedSizesEditor({
  value,
  onChange,
}: {
  value: SizesMap | undefined;
  onChange: (v: SizesMap | undefined) => void;
}) {
  const [collapsedTiers, setCollapsedTiers] = useState<Set<string>>(
    () => new Set(Object.keys(value || {}).slice(1)),
  );
  const [newTierName, setNewTierName] = useState("");

  const sizes = value || {};
  const tierNames = Object.keys(sizes);

  const toggleCollapse = (tier: string) => {
    setCollapsedTiers((prev) => {
      const next = new Set(prev);
      if (next.has(tier)) {
        next.delete(tier);
      } else {
        next.add(tier);
      }
      return next;
    });
  };

  const addTier = (tierName: string) => {
    const name = tierName.trim();
    if (!name || sizes[name]) {
      return;
    }
    const next = { ...sizes, [name]: {} };
    onChange(next);
    setCollapsedTiers((prev) => {
      const collapsed = new Set(prev);
      collapsed.delete(name);
      return collapsed;
    });
    setNewTierName("");
  };

  const removeTier = (tier: string) => {
    const next = { ...sizes };
    delete next[tier];
    onChange(Object.keys(next).length > 0 ? next : undefined);
  };

  const addRatio = (tier: string, ratio: string) => {
    if (!ratio.trim()) {
      return;
    }
    const next = { ...sizes, [tier]: { ...sizes[tier], [ratio.trim()]: "" } };
    onChange(next);
  };

  const updateResolution = (
    tier: string,
    ratio: string,
    resolution: string,
  ) => {
    const next = { ...sizes, [tier]: { ...sizes[tier], [ratio]: resolution } };
    onChange(next);
  };

  const removeRatio = (tier: string, ratio: string) => {
    const tierData = { ...sizes[tier] };
    delete tierData[ratio];
    const next = { ...sizes, [tier]: tierData };
    onChange(next);
  };

  const fillCommonRatios = (tier: string) => {
    const tierData = { ...sizes[tier] };
    COMMON_ASPECT_RATIOS.forEach((r) => {
      if (!(r in tierData)) {
        tierData[r] = "";
      }
    });
    onChange({ ...sizes, [tier]: tierData });
  };

  const availableTierChips = COMMON_TIERS.filter((t) => !sizes[t]);

  return (
    <div className="space-y-3">
      <label className="text-xs text-muted-foreground">
        支持的尺寸 (supportedSizes)
      </label>

      {tierNames.map((tier) => {
        const isCollapsed = collapsedTiers.has(tier);
        const ratios = sizes[tier] || {};
        const ratioEntries = Object.entries(ratios);

        return (
          <div
            key={tier}
            className="rounded-lg border border-border/20 bg-background/70"
          >
            <div
              className={cn(
                "flex min-h-10 items-center justify-between gap-3 px-3",
                !isCollapsed && "border-b border-border/20",
              )}
            >
              <button
                type="button"
                onClick={() => toggleCollapse(tier)}
                className="flex min-w-0 items-center gap-2 rounded-md text-sm font-medium text-foreground/80 outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50"
              >
                <ChevronRight
                  className={cn(
                    "size-4 shrink-0 transition-transform duration-200",
                    !isCollapsed && "rotate-90",
                  )}
                />
                <span className="truncate">{tier}</span>
                <span className="shrink-0 text-xs font-normal text-muted-foreground">
                  {ratioEntries.length} 项
                </span>
              </button>
              <div className="flex shrink-0 items-center gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="xs"
                  onClick={() => fillCommonRatios(tier)}
                  title="填充常见比例"
                >
                  补齐常见比例
                </Button>
                <Button
                  type="button"
                  variant="destructive-ghost"
                  size="icon-xs"
                  onClick={() => removeTier(tier)}
                  aria-label={`删除 ${tier} 档位`}
                  title="删除档位"
                >
                  <Trash2 />
                </Button>
              </div>
            </div>

            {!isCollapsed && (
              <div className="space-y-2 p-3">
                {ratioEntries.length > 0 ? (
                  <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                    {ratioEntries.map(([ratio, resolution]) => (
                      <div
                        key={ratio}
                        className="group flex min-w-0 items-center gap-2 rounded-lg border border-border/20 bg-background/70 p-2 transition-colors focus-within:border-border/50"
                      >
                        <span className="w-10 shrink-0 text-right font-mono text-xs text-muted-foreground">
                          {ratio}
                        </span>
                        <Input
                          aria-label={`${tier} ${ratio} 尺寸`}
                          placeholder="1024x1024"
                          value={resolution}
                          onChange={(e) =>
                            updateResolution(tier, ratio, e.target.value)
                          }
                          className="min-w-0 flex-1 font-mono text-xs"
                        />
                        <Button
                          type="button"
                          variant="destructive-ghost"
                          size="icon-xs"
                          onClick={() => removeRatio(tier, ratio)}
                          aria-label={`删除 ${ratio} 比例`}
                          title={`删除 ${ratio}`}
                        >
                          <X />
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-lg border border-dashed border-border/40 py-4 text-center text-xs text-muted-foreground">
                    暂无尺寸比例
                  </div>
                )}

                {(() => {
                  const existingRatios = new Set(Object.keys(ratios));
                  const available = COMMON_ASPECT_RATIOS.filter(
                    (r) => !existingRatios.has(r),
                  );
                  if (available.length === 0) {
                    return null;
                  }
                  return (
                    <div className="flex flex-wrap items-center gap-2 border-t border-border/20 pt-2">
                      <span className="text-xs text-muted-foreground">
                        添加比例
                      </span>
                      {available.map((r) => (
                        <Button
                          key={r}
                          type="button"
                          variant="ghost"
                          size="xs"
                          onClick={() => addRatio(tier, r)}
                        >
                          + {r}
                        </Button>
                      ))}
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        );
      })}

      <div className="flex flex-wrap items-center gap-2">
        {availableTierChips.map((t) => (
          <Button
            key={t}
            type="button"
            variant="outline"
            size="xs"
            onClick={() => addTier(t)}
          >
            + {t}
          </Button>
        ))}
        <div className="ml-auto flex items-center gap-2">
          <Input
            placeholder="自定义档位"
            value={newTierName}
            onChange={(e) => setNewTierName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addTier(newTierName)}
            className="h-8 w-28 font-mono text-xs"
          />
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => addTier(newTierName)}
            disabled={!newTierName.trim()}
          >
            添加
          </Button>
        </div>
      </div>
    </div>
  );
}
