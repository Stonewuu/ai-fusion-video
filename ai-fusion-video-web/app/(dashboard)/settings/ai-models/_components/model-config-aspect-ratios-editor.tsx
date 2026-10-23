"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Check, X } from "lucide-react";
import { COMMON_ASPECT_RATIOS } from "./model-config-options";

export function AspectRatiosEditor({
  value,
  onChange,
  label,
  hint,
  presetOptions,
}: {
  value: string[] | undefined;
  onChange: (v: string[] | undefined) => void;
  label?: string;
  hint?: string;
  presetOptions?: string[];
}) {
  const [customRatio, setCustomRatio] = useState("");
  const selected = new Set(value || []);
  const commonOptions = presetOptions || COMMON_ASPECT_RATIOS;
  const selectedValues = Array.from(selected);
  const availableOptions = commonOptions.filter(
    (ratio) => !selected.has(ratio),
  );

  const toggle = (ratio: string) => {
    const next = new Set(selected);
    if (next.has(ratio)) {
      next.delete(ratio);
    } else {
      next.add(ratio);
    }
    const arr = Array.from(next);
    onChange(arr.length > 0 ? arr : undefined);
  };

  const addCustom = () => {
    const r = customRatio.trim();
    if (!r || selected.has(r)) {
      return;
    }
    onChange([...Array.from(selected), r]);
    setCustomRatio("");
  };

  return (
    <div className="flex min-w-0 flex-col gap-3">
      <div className="flex flex-col gap-1">
        <label className="text-xs text-muted-foreground">
          {label || "支持的比例 (supportedAspectRatios)"}
        </label>
        {hint && <p className="text-xs text-muted-foreground/70">{hint}</p>}
      </div>

      {selectedValues.length > 0 ? (
        <div className="flex min-w-0 flex-wrap gap-2 p-1">
          {selectedValues.map((ratio) => (
            <div
              key={ratio}
              className="group flex min-h-9 max-w-full items-center gap-2 rounded-lg border border-border/20 bg-background/70 py-1 pl-2 pr-1 transition-colors focus-within:border-border/50"
            >
              <Check className="size-4 shrink-0 text-primary" />
              <span className="min-w-0 break-words text-sm text-foreground/80">
                {ratio === "adaptive" ? "自适应" : ratio === "auto" ? "自动" : ratio}
              </span>
              <Button
                type="button"
                variant="destructive-ghost"
                size="icon-xs"
                onClick={() => toggle(ratio)}
                aria-label={`删除 ${ratio}`}
                title={`删除 ${ratio}`}
              >
                <X />
              </Button>
            </div>
          ))}
        </div>
      ) : (
        <div className="rounded-lg border border-dashed border-border/40 py-4 text-center text-xs text-muted-foreground">
          暂无已选项
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 border-t border-border/20 pt-2">
        {availableOptions.length > 0 && (
          <>
            <span className="text-xs text-muted-foreground">添加选项</span>
            {availableOptions.map((ratio) => (
              <Button
                key={ratio}
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => toggle(ratio)}
              >
                + {ratio}
              </Button>
            ))}
          </>
        )}
        <div className="flex w-full min-w-0 items-center gap-2 p-1 sm:ml-auto sm:w-auto">
          <Input
            aria-label={`${label || "比例"}自定义选项`}
            placeholder={label?.includes("分辨率") ? "自定义分辨率，如：1440p" : "自定义比例，如：16:9"}
            value={customRatio}
            onChange={(e) => setCustomRatio(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                addCustom();
              }
            }}
            className="min-w-0 flex-1 sm:w-52"
          />
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={addCustom}
            disabled={!customRatio.trim() || selected.has(customRatio.trim())}
          >
            添加
          </Button>
        </div>
      </div>
    </div>
  );
}
