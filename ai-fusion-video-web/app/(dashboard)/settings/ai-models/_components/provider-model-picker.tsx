"use client";

import { useState } from "react";
import { ModelVendorIcon } from "@/components/dashboard/model-vendor-icon";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
  const types = [1, 2, 3].filter((value) =>
    models.some((model) => model.modelType === value),
  );
  const filtered = models.filter(
    (model) =>
      (!type || type === model.modelType) &&
      `${model.id} ${model.name}`
        .toLowerCase()
        .includes(query.trim().toLowerCase()),
  );
  const selectable = filtered.filter(
    (model) => !model.configurationHint && !existing.has(model.id),
  );
  const allSelected =
    selectable.length > 0 &&
    selectable.every((model) => selected.has(model.id));
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
    <div className="flex min-w-0 flex-col gap-4">
      <Input
        aria-label="搜索可用模型"
        placeholder="搜索模型名称或代码"
        value={query}
        disabled={busy}
        onChange={(event) => setQuery(event.target.value)}
      />
      <div
        className="flex flex-wrap items-center gap-2"
        role="group"
        aria-label="模型类型"
      >
        {[0, ...types].map((value) => (
          <Button
            key={value}
            type="button"
            variant={type === value ? "secondary" : "ghost"}
            aria-pressed={type === value}
            disabled={busy}
            onClick={() => setType(value)}
          >
            {value === 0 ? "全部" : MODEL_TYPE_LABELS[value]}
          </Button>
        ))}
        <Button
          type="button"
          variant="ghost"
          className="ml-auto"
          disabled={busy || !selectable.length}
          onClick={toggleVisible}
        >
          {allSelected ? "取消全选" : "选择当前列表"}
        </Button>
      </div>
      <div className="flex min-w-0 flex-col gap-1">
        {filtered.length === 0 && (
          <p className="p-4 text-sm text-muted-foreground" role="status">
            没有匹配的模型。
          </p>
        )}
        {filtered.map((model) => {
          const added = existing.has(model.id);
          const disabled = busy || added || !!model.configurationHint;
          return (
            <div
              key={model.id}
              className="flex min-w-0 items-start gap-3 rounded-lg px-2 py-3 hover:bg-muted/50"
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
                <span className="break-words text-sm font-medium">
                  {model.name}
                </span>
                <span className="break-all text-xs text-muted-foreground">
                  {model.id}
                </span>
                {model.configurationHint && (
                  <span className="text-xs text-muted-foreground">
                    {model.configurationHint}
                  </span>
                )}
              </span>
              <span className="shrink-0 text-xs text-muted-foreground">
                {added ? "已添加" : MODEL_TYPE_LABELS[model.modelType]}
              </span>
              {!added && model.configurationHint && (
                <Button
                  type="button"
                  variant="outline"
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
