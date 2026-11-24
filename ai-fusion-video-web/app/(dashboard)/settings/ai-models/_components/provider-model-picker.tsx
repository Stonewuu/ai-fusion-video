"use client";

import { useState } from "react";
import { Search, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { ModelVendorIcon } from "@/components/dashboard/model-vendor-icon";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { MODEL_TYPE_LABELS } from "@/lib/api/ai-model";
import type { ProviderModelCandidate } from "@/lib/api/provider-catalog";

interface ProviderModelPickerProps {
  models: ProviderModelCandidate[];
  selected: Set<string>;
  existing: Set<string>;
  busy: boolean;
  onChange: (ids: Set<string>) => void;
  onConfigure: (model: ProviderModelCandidate) => void;
}

export function ProviderModelPicker({
  models,
  selected,
  existing,
  busy,
  onChange,
  onConfigure,
}: ProviderModelPickerProps) {
  const [query, setQuery] = useState("");
  const [type, setType] = useState(0);
  const [hideExisting, setHideExisting] = useState(false);

  const types = [1, 2, 3].filter((value) =>
    models.some((model) => model.modelType === value),
  );
  const filtered = models.filter((model) => {
    if (hideExisting && existing.has(model.id)) {
      return false;
    }
    if (type !== 0 && type !== model.modelType) {
      return false;
    }
    const key = query.trim().toLowerCase();
    if (!key) {
      return true;
    }
    return `${model.id} ${model.name}`.toLowerCase().includes(key);
  });
  const selectable = filtered.filter(
    (model) => !model.configurationHint && !existing.has(model.id),
  );
  const allSelected =
    selectable.length > 0
    && selectable.every((model) => selected.has(model.id));

  const toggle = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    onChange(next);
  };

  const toggleVisible = () => {
    const next = new Set(selected);
    for (const model of selectable) {
      if (allSelected) {
        next.delete(model.id);
      } else {
        next.add(model.id);
      }
    }
    onChange(next);
  };

  return (
    <div className="flex min-w-0 flex-col gap-3">
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          aria-label="搜索可用模型"
          placeholder="搜索模型名称或代码"
          value={query}
          disabled={busy}
          onChange={(event) => setQuery(event.target.value)}
          className={cn("pl-9", query && "pr-9")}
        />
        {query && (
          <button
            type="button"
            aria-label="清空搜索"
            title="清空搜索"
            onClick={() => setQuery("")}
            className="absolute right-2.5 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
          >
            <X className="size-3.5" />
          </button>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <div
            className="inline-flex h-9 items-center rounded-4xl bg-muted p-[3px] ring-1 ring-inset ring-border/35"
            role="group"
            aria-label="模型类型"
          >
            {[0, ...types].map((value) => {
              const active = type === value;
              const label =
                value === 0 ? "全部" : MODEL_TYPE_LABELS[value].replace("生成", "");
              return (
                <button
                  key={value}
                  type="button"
                  aria-pressed={active}
                  disabled={busy}
                  onClick={() => setType(value)}
                  className={cn(
                    "flex h-full items-center rounded-4xl px-3 text-xs font-medium outline-none transition-colors motion-reduce:transition-none focus-visible:ring-2 focus-visible:ring-ring",
                    active
                      ? "bg-background text-foreground shadow-xs ring-1 ring-border/25 dark:bg-card"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {label}
                </button>
              );
            })}
          </div>

          {existing.size > 0 && (
            <div className="flex items-center gap-2 px-1">
              <Switch
                id="provider-picker-hide-existing"
                checked={hideExisting}
                disabled={busy}
                onCheckedChange={setHideExisting}
              />
              <Label
                htmlFor="provider-picker-hide-existing"
                className="cursor-pointer text-xs text-muted-foreground"
              >
                隐藏已添加
              </Label>
            </div>
          )}
        </div>

        <Button
          type="button"
          variant="ghost"
          size="sm"
          disabled={busy || !selectable.length}
          onClick={toggleVisible}
        >
          {allSelected ? "取消全选" : "选择当前列表"}
        </Button>
      </div>

      <div className="flex min-w-0 flex-col gap-1">
        {filtered.length === 0 && (
          <p className="p-4 text-center text-sm text-muted-foreground" role="status">
            没有匹配的模型。
          </p>
        )}
        {filtered.map((model) => {
          const added = existing.has(model.id);
          const disabled = busy || added || !!model.configurationHint;
          return (
            <div
              key={model.id}
              className={cn(
                "flex min-w-0 items-start gap-3 rounded-lg px-2.5 py-2.5 transition-colors hover:bg-muted/45",
                added && "opacity-65",
              )}
            >
              <Checkbox
                checked={added || selected.has(model.id)}
                disabled={disabled}
                aria-label={`添加 ${model.name}`}
                onCheckedChange={() => toggle(model.id)}
              />
              <ModelVendorIcon
                source={{
                  name: model.name,
                  code: model.id,
                  capabilityPresetCode: model.capabilityPresetCode,
                }}
              />
              <span className="flex min-w-0 flex-1 flex-col gap-1">
                <span className="break-words text-sm font-medium text-foreground">
                  {model.name}
                </span>
                <span className="break-all font-mono text-xs text-muted-foreground">
                  {model.id}
                </span>
                {model.configurationHint && (
                  <span className="text-xs text-muted-foreground">
                    {model.configurationHint}
                  </span>
                )}
              </span>
              <Badge variant={added ? "outline" : "secondary"} className="shrink-0 text-xs">
                {added ? "已添加" : MODEL_TYPE_LABELS[model.modelType]}
              </Badge>
              {!added && model.configurationHint && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={busy}
                  onClick={() => onConfigure(model)}
                >
                  配置
                </Button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
