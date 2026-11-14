"use client";

import { useState } from "react";
import { ChevronRight, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import type { ProviderDefinition } from "@/lib/api/provider-catalog";
import { ProviderPresetList } from "./provider-preset-list";
import {
  DEFAULT_PROVIDER_PRESET_LIMIT,
  orderProviderPresets,
} from "./provider-preset-order";

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
  const [showAll, setShowAll] = useState(false);
  const orderedProviders = orderProviderPresets(providers);

  const selectFromDialog = (provider: ProviderDefinition) => {
    setShowAll(false);
    onSelect(provider);
  };

  return (
    <Dialog open={showAll} onOpenChange={setShowAll}>
      <section className="flex min-w-0 flex-col gap-3 rounded-xl border border-border/30 bg-muted/15 p-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 flex-col gap-1">
            <h3 className="text-xs font-medium">常用提供商</h3>
            <p className="text-xs text-muted-foreground">
              选择平台自动填写地址；赞助商可快速配置模型。
            </p>
          </div>
          {orderedProviders.length > DEFAULT_PROVIDER_PRESET_LIMIT && (
            <DialogTrigger
              render={
                <Button
                  type="button"
                  variant="ghost"
                  disabled={disabled || loading}
                />
              }
            >
              更多
              <ChevronRight />
            </DialogTrigger>
          )}
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
              type="button"
              variant="outline"
              disabled={disabled || loading}
              onClick={onReload}
            >
              重试
            </Button>
          </div>
        )}
        <ProviderPresetList
          providers={orderedProviders.slice(0, DEFAULT_PROVIDER_PRESET_LIMIT)}
          selectedId={selectedId}
          disabled={disabled}
          allowQuickSetup={allowQuickSetup}
          onSelect={onSelect}
        />
      </section>
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-3xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>全部提供商</DialogTitle>
          <DialogDescription>
            选择平台自动填写地址；赞助商可快速配置模型。
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody>
          <ProviderPresetList
            providers={orderedProviders}
            selectedId={selectedId}
            disabled={disabled}
            allowQuickSetup={allowQuickSetup}
            onSelect={selectFromDialog}
          />
        </DialogScrollBody>
      </DialogContent>
    </Dialog>
  );
}
