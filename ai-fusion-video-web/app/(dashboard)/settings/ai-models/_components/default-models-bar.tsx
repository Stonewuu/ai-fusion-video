"use client";

import {
  AlertTriangle,
  ArrowUpRight,
  AudioLines,
  ImageIcon,
  Loader2,
  MessageSquareText,
  Star,
  Video,
} from "lucide-react";
import { getModelDisplayParts } from "@/lib/model-display";
import type { AiModel } from "@/lib/api/ai-model";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ModelVendorIcon } from "@/components/dashboard/model-vendor-icon";
import type { DefaultModelSlot } from "./ai-models-filter";

const MODEL_TYPE_ICONS: Record<number, typeof MessageSquareText> = {
  1: MessageSquareText,
  2: ImageIcon,
  3: Video,
  4: AudioLines,
};

interface DefaultModelsBarProps {
  slots: DefaultModelSlot[];
  updatingModelType: number | null;
  onSelectDefault: (model: AiModel) => Promise<void>;
  onLocateConfig: (configId: number) => void;
}

export function DefaultModelsBar({
  slots,
  updatingModelType,
  onSelectDefault,
  onLocateConfig,
}: DefaultModelsBarProps) {
  return (
    <section
      aria-label="全局默认模型"
      className="mb-6 rounded-xl border border-border/30 bg-card/50 p-4 backdrop-blur-sm"
    >
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Star className="size-4 text-amber-500" />
          <h2 className="text-sm font-semibold text-foreground">默认模型</h2>
        </div>
        <p className="text-xs text-muted-foreground">
          助手与画布按能力类型默认调用对应模型。
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {slots.map((slot) => {
          const TypeIcon = MODEL_TYPE_ICONS[slot.modelType] ?? MessageSquareText;
          const isUpdating = updatingModelType === slot.modelType;
          const selectItems = slot.candidates.map(({ model }) => ({
            value: String(model.id),
            label: getModelDisplayParts(model).name,
          }));
          const activeValidValue =
            slot.model && !slot.unavailableReason ? String(slot.model.id) : "";
          const activeTooltip =
            slot.model && !slot.unavailableReason
              ? `${getModelDisplayParts(slot.model).name}${slot.config ? ` · ${slot.config.name}` : ""}`
              : undefined;

          return (
            <div
              key={slot.modelType}
              className="flex min-w-0 flex-col gap-2 rounded-lg border border-border/20 bg-background/70 p-3"
            >
              <div className="flex h-6 min-w-0 items-center justify-between gap-2">
                <span className="inline-flex shrink-0 items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  <TypeIcon className="size-3.5" />
                  {slot.label}
                </span>
                {slot.config && (
                  <div className="flex min-w-0 items-center gap-1">
                    <span
                      className="truncate text-xs text-muted-foreground"
                      title={slot.config.name}
                    >
                      {slot.config.name}
                    </span>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-xs"
                      aria-label={`定位到 ${slot.config.name}`}
                      title={`定位到所属服务：${slot.config.name}`}
                      onClick={() => onLocateConfig(slot.config!.id)}
                    >
                      <ArrowUpRight />
                    </Button>
                  </div>
                )}
              </div>

              {slot.candidates.length > 0 ? (
                <Select
                  value={activeValidValue || undefined}
                  disabled={isUpdating}
                  onValueChange={(value) => {
                    const targetId = Number(value);
                    const matched = slot.candidates.find(
                      (item) => item.model.id === targetId,
                    );
                    if (matched && matched.model.id !== slot.model?.id) {
                      void onSelectDefault(matched.model);
                    }
                  }}
                  items={selectItems}
                >
                  <SelectTrigger
                    aria-label={`选择默认${slot.label}模型`}
                    title={activeTooltip}
                    className="w-full overflow-hidden text-xs"
                  >
                    <span className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
                      {isUpdating ? (
                        <Loader2 className="size-3.5 shrink-0 animate-spin text-muted-foreground" />
                      ) : slot.model && !slot.unavailableReason ? (
                        <ModelVendorIcon source={slot.model} className="size-3.5 shrink-0" />
                      ) : null}
                      <SelectValue
                        placeholder="未设置默认模型"
                        className="block min-w-0 flex-1 truncate"
                      />
                    </span>
                  </SelectTrigger>
                  <SelectContent className="text-xs">
                    <SelectGroup>
                      {slot.candidates.map(({ model, config }) => (
                        <SelectItem
                          key={model.id}
                          value={String(model.id)}
                          className="text-xs"
                        >
                          <ModelVendorIcon source={model} className="size-3.5 shrink-0" />
                          <span className="min-w-0 truncate">
                            {getModelDisplayParts(model).name}
                          </span>
                          <span className="ml-auto shrink-0 text-muted-foreground">
                            {config.name}
                          </span>
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              ) : (
                <div className="flex h-9 items-center rounded-4xl border border-dashed border-border/40 px-3 text-xs text-muted-foreground">
                  暂无可用模型
                </div>
              )}

              {slot.unavailableReason && slot.model && (
                <p className="flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400">
                  <AlertTriangle className="size-3.5 shrink-0" />
                  <span className="truncate">
                    原默认「{getModelDisplayParts(slot.model).name}」{slot.unavailableReason}
                  </span>
                </p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
