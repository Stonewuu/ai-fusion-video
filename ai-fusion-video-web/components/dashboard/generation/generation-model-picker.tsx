"use client";

import { Cpu, Loader2 } from "lucide-react";
import type { AiModel } from "@/lib/api/ai-model";
import { getModelDisplayParts } from "@/lib/model-display";
import { ModelVendorIcon } from "@/components/dashboard/model-vendor-icon";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface GenerationModelPickerProps {
  models: AiModel[];
  modelId?: number;
  loading?: boolean;
  compact?: boolean;
  className?: string;
  onChange: (modelId: number) => void;
}

/** 生成页模型选择器，始终保留可点击的模型入口。 */
export function GenerationModelPicker({
  models,
  modelId,
  loading = false,
  compact = false,
  className,
  onChange,
}: GenerationModelPickerProps) {
  const selectedModel = models.find((model) => model.id === modelId) ?? models[0];
  const shellClass = compact
    ? "h-8 max-w-[min(15rem,42vw)] rounded-xl px-2.5 text-xs"
    : "h-10 w-full rounded-xl px-3 text-sm";

  if (loading) {
    return (
      <div
        className={cn(
          "flex min-w-0 items-center gap-2 text-muted-foreground",
          shellClass,
          className,
        )}
        aria-live="polite"
      >
        <Loader2 className="size-3.5 shrink-0 animate-spin" />
        <span className="truncate">加载模型…</span>
      </div>
    );
  }

  if (!selectedModel) {
    return (
      <div
        className={cn(
          "flex min-w-0 items-center gap-2 text-muted-foreground",
          shellClass,
          className,
        )}
      >
        <Cpu className="size-3.5 shrink-0" />
        <span className="truncate">暂无可用模型</span>
      </div>
    );
  }

  return (
    <Select
      value={selectedModel.id}
      onValueChange={(value) => onChange(Number(value))}
      items={models.map((model) => ({
        value: model.id,
        label: getModelDisplayParts(model).name,
      }))}
    >
      <SelectTrigger
        aria-label="选择生成模型"
        className={cn(
          "min-w-0 border-transparent bg-transparent hover:bg-background/80",
          shellClass,
          className,
        )}
      >
        <ModelVendorIcon source={selectedModel} className="size-3.5" />
        <SelectValue placeholder="选择模型" />
      </SelectTrigger>
      <SelectContent
        side={compact ? "top" : "bottom"}
        sideOffset={compact ? 8 : 4}
        align="start"
        alignItemWithTrigger={false}
        className={cn(
          "border border-border/55 bg-popover/95 p-1.5 shadow-[0_18px_55px_-22px_rgba(15,23,42,.38)]",
          compact
            ? "w-[min(17rem,calc(100vw-2rem))] min-w-[13rem]"
            : "w-[min(22rem,calc(100vw-2rem))] min-w-[16rem]",
        )}
      >
        <SelectGroup className="p-0">
          <SelectLabel className="px-2.5 py-1.5 text-[10px] font-medium">
            生成模型
          </SelectLabel>
          {models.map((model) => {
            const display = getModelDisplayParts(model);
            const modelCode = model.code.trim();
            return (
              <SelectItem
                key={model.id}
                value={model.id}
                title={modelCode ? `${display.name} · ${modelCode}` : display.name}
                className="min-h-11 py-2.5 pl-2.5"
              >
                <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-muted/70">
                  <ModelVendorIcon source={model} className="size-4" />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-xs font-medium">
                    {display.name}
                  </span>
                  {modelCode && (
                    <span className="mt-0.5 block truncate font-mono text-[10px] text-muted-foreground">
                      {modelCode}
                    </span>
                  )}
                </span>
              </SelectItem>
            );
          })}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
