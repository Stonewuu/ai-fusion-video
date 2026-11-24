"use client";

import { ArrowRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ProviderVendorIcon } from "@/components/dashboard/model-vendor-icon";
import type { ProviderDefinition } from "@/lib/api/provider-catalog";
import { cn } from "@/lib/utils";
import { SponsorNotice } from "./sponsor-notice";

interface ProviderPresetListProps {
  providers: ProviderDefinition[];
  selectedId?: string;
  disabled: boolean;
  allowQuickSetup: boolean;
  onSelect: (provider: ProviderDefinition) => void;
}

export function ProviderPresetList({
  providers,
  selectedId,
  disabled,
  allowQuickSetup,
  onSelect,
}: ProviderPresetListProps) {
  return (
    <div className="grid min-w-0 gap-2.5 sm:grid-cols-2">
      {providers.map((provider) => {
        const selected = selectedId === provider.id;
        const isQuickFlow = Boolean(provider.quickSetup && allowQuickSetup);
        const surface = cn(
          "flex min-w-0 flex-col justify-between gap-2 rounded-xl border p-3 text-left transition-colors motion-reduce:transition-none",
          selected
            ? "border-primary/50 bg-primary/8"
            : "border-border/30 bg-background/70",
        );

        const icon = (
          <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-muted/60">
            <ProviderVendorIcon
              provider={provider.vendor ?? provider.id}
              className="size-5"
            />
          </span>
        );

        // 赞助商卡片含外部注册链接，需保持独立按钮避免按钮内嵌 <a>
        if (provider.sponsored) {
          return (
            <div key={provider.id} className={surface}>
              <div className="flex items-center gap-2.5">
                {icon}
                <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
                  <span className="text-sm font-medium text-foreground">
                    {provider.label}
                  </span>
                  <Badge variant="secondary" className="bg-primary/10 text-primary">
                    赞助商
                  </Badge>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={disabled}
                  aria-label={`${isQuickFlow ? "快速配置" : "选用"} ${provider.label}`}
                  onClick={() => onSelect(provider)}
                >
                  {isQuickFlow ? "快速配置" : "选用"}
                  {isQuickFlow && <ArrowRight />}
                </Button>
              </div>
              <div className="min-w-0">
                <SponsorNotice provider={provider} />
              </div>
            </div>
          );
        }

        return (
          <button
            key={provider.id}
            type="button"
            disabled={disabled}
            aria-pressed={selected}
            onClick={() => onSelect(provider)}
            className={cn(
              surface,
              "outline-none hover:border-primary/40 hover:bg-muted/35 focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50",
            )}
          >
            <div className="flex w-full items-center gap-2.5">
              {icon}
              <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
                <span className="text-sm font-medium text-foreground">
                  {provider.label}
                </span>
                {isQuickFlow && (
                  <Badge variant="secondary">快速向导</Badge>
                )}
              </div>
              {isQuickFlow && (
                <ArrowRight className="size-4 shrink-0 text-muted-foreground" />
              )}
            </div>
            <span className="block w-full break-all text-xs leading-relaxed text-muted-foreground">
              {provider.hint ?? provider.url}
            </span>
          </button>
        );
      })}
    </div>
  );
}
