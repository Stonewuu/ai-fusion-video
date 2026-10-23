"use client";

import { Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ProviderVendorIcon } from "@/components/dashboard/model-vendor-icon";
import type { ProviderDefinition } from "@/lib/api/provider-catalog";
import { cn } from "@/lib/utils";
import { SponsorNotice } from "./sponsor-notice";

interface ProviderPresetPickerProps {
  providers: ProviderDefinition[];
  selectedId?: string;
  loading: boolean;
  error: string | null;
  disabled: boolean;
  allowQuickSetup: boolean;
  onReload: () => void;
  onSelect: (provider: ProviderDefinition) => void;
}

export function ProviderPresetPicker({
  providers,
  selectedId,
  loading,
  error,
  disabled,
  allowQuickSetup,
  onReload,
  onSelect,
}: ProviderPresetPickerProps) {
  return (
    <section className="flex min-w-0 flex-col gap-3 rounded-xl border border-border/30 bg-muted/15 p-3">
      <div className="flex flex-col gap-1">
        <h3 className="text-xs font-medium">常用提供商</h3>
        <p className="text-xs text-muted-foreground">
          选择平台自动填写地址；赞助商可快速配置模型。
        </p>
      </div>
      {loading && (
        <p
          className="flex items-center gap-2 text-xs text-muted-foreground"
          role="status"
        >
          <Loader2 className="size-4 animate-spin" />
          加载提供商…
        </p>
      )}
      {error && (
        <div className="flex items-center justify-between gap-2">
          <p role="alert" className="text-xs text-destructive">
            {error}
          </p>
          <Button
            variant="outline"
            size="sm"
            disabled={disabled || loading}
            onClick={onReload}
          >
            重试
          </Button>
        </div>
      )}
      <div className="grid min-w-0 gap-2 sm:grid-cols-2">
        {providers.map((provider) => {
          const selected = selectedId === provider.id;
          const surface = cn(
            "min-w-0 rounded-xl border p-3 text-left transition-colors",
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
                    <span className="text-xs font-medium">
                      {provider.label}
                    </span>
                    {provider.sponsored && (
                      <Badge variant="secondary">赞助商</Badge>
                    )}
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
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
                <span className="block text-xs font-medium">
                  {provider.label}
                </span>
                <span className="mt-1 block break-all text-xs text-muted-foreground">
                  {provider.hint ?? provider.url}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
