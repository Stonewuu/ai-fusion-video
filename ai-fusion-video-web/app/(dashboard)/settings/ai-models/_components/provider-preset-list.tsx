"use client";

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
    <div className="grid min-w-0 gap-2 sm:grid-cols-2">
      {providers.map((provider) => {
        const selected = selectedId === provider.id;
        const surface = cn(
          "min-w-0 rounded-xl border p-3 text-left transition-colors motion-reduce:transition-none",
          selected
            ? "border-primary/50 bg-primary/8"
            : "border-border/40 bg-background/50",
        );
        const icon = (
          <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-muted/60">
            <ProviderVendorIcon
              provider={provider.vendor ?? provider.id}
              className="size-5"
            />
          </span>
        );
        if (provider.sponsored || provider.quickSetup) {
          return (
            <div key={provider.id} className={surface}>
              <div className="flex items-center gap-2">
                {icon}
                <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
                  <span className="text-xs font-medium">{provider.label}</span>
                  {provider.sponsored && (
                    <Badge variant="secondary">赞助商</Badge>
                  )}
                </div>
                <Button
                  type="button"
                  variant="outline"
                  disabled={disabled}
                  aria-label={`配置 ${provider.label}`}
                  onClick={() => onSelect(provider)}
                >
                  {provider.quickSetup && allowQuickSetup ? "配置" : "选用"}
                </Button>
              </div>
              <div className="mt-2 min-w-0">
                {provider.sponsored ? (
                  <SponsorNotice provider={provider} />
                ) : (
                  <p className="break-all text-xs text-muted-foreground">
                    {provider.hint ?? provider.url}
                  </p>
                )}
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
              "flex items-center gap-2 outline-none hover:border-primary/40 hover:bg-muted/30 focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50",
            )}
          >
            {icon}
            <span className="min-w-0">
              <span className="block text-xs font-medium">{provider.label}</span>
              <span className="mt-1 block break-all text-xs text-muted-foreground">
                {provider.hint ?? provider.url}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}
