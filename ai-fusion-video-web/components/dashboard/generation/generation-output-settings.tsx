"use client";

import { Check, Minus, Plus } from "lucide-react";
import type { GenerationCapabilities } from "@/lib/generation-capabilities";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { GenerationDurationSlider } from "./generation-duration-slider";
import type { GenerationFormState, WorkbenchMode } from "./generation-types";
import { parseLimit } from "./generation-utils";

interface GenerationOutputSettingsProps {
  mode: WorkbenchMode;
  capabilities: GenerationCapabilities | null;
  form: GenerationFormState;
  onChange: (patch: Partial<GenerationFormState>) => void;
}

function ChoiceGroup({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  if (options.length === 0) return null;

  return (
    <div>
      <p className="text-[10px] font-medium text-muted-foreground">{label}</p>
      <div className="mt-1.5 flex flex-wrap gap-1.5">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            aria-pressed={value === option}
            className={cn(
              "h-7 rounded-lg border px-2.5 text-[11px] font-medium transition-all",
              value === option
                ? "border-primary/30 bg-primary/[0.08] text-primary shadow-sm"
                : "border-border/40 bg-background/65 text-muted-foreground hover:border-primary/20 hover:bg-background hover:text-foreground",
            )}
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
}

function Stepper({
  label,
  value,
  min,
  max,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  suffix: string;
  onChange: (value: number) => void;
}) {
  return (
    <div className="flex h-11 items-center justify-between gap-3 rounded-xl border border-border/40 bg-background/55 px-3">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <div className="flex items-center gap-1">
        <button
          type="button"
          disabled={value <= min}
          onClick={() => onChange(Math.max(min, value - 1))}
          className="grid h-7 w-7 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-30"
          aria-label={`减少${label}`}
        >
          <Minus className="h-3.5 w-3.5" />
        </button>
        <span className="inline-flex min-w-10 justify-center text-sm font-semibold tabular-nums">
          {value}
          <span className="ml-0.5 text-[10px] font-normal text-muted-foreground">
            {suffix}
          </span>
        </span>
        <button
          type="button"
          disabled={value >= max}
          onClick={() => onChange(Math.min(max, value + 1))}
          className="grid h-7 w-7 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-30"
          aria-label={`增加${label}`}
        >
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}

function ToggleOption({
  label,
  checked,
  disabled,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "flex h-10 items-center justify-between gap-3 rounded-xl border px-3 text-xs transition-all",
        checked
          ? "border-primary/25 bg-primary/[0.065] text-foreground"
          : "border-border/40 bg-background/55 text-muted-foreground hover:bg-background hover:text-foreground",
        disabled && "cursor-not-allowed opacity-35",
      )}
    >
      <span>{label}</span>
      <span
        className={cn(
          "grid h-5 w-5 place-items-center rounded-md border",
          checked
            ? "border-primary bg-primary text-primary-foreground"
            : "border-border bg-background",
        )}
      >
        {checked && <Check className="h-3 w-3" />}
      </span>
    </button>
  );
}

export function GenerationOutputSettings({
  mode,
  capabilities,
  form,
  onChange,
}: GenerationOutputSettingsProps) {
  const outputLimit = parseLimit(capabilities?.maxCount || 0, 4);

  return (
    <section className="rounded-[18px] border border-border/45 bg-background/48 p-3.5">
      <div className="flex items-center justify-between gap-3">
        <h4 className="text-xs font-semibold">输出参数</h4>
        <span className="text-[10px] text-muted-foreground">仅显示模型支持项</span>
      </div>

      <div className="mt-3 space-y-3.5">
        <ChoiceGroup
          label="画面比例"
          value={form.ratio}
          options={capabilities?.supportedAspectRatios || []}
          onChange={(ratio) => onChange({ ratio })}
        />
        <ChoiceGroup
          label="分辨率"
          value={form.resolution}
          options={capabilities?.supportedResolutions || []}
          onChange={(resolution) => onChange({ resolution })}
        />

        <div className="grid gap-2 sm:grid-cols-2">
          <Stepper
            label="输出数量"
            value={form.count}
            min={1}
            max={outputLimit}
            suffix="个"
            onChange={(count) => onChange({ count })}
          />
          {mode === "video" && capabilities && (
            <GenerationDurationSlider
              value={form.duration}
              min={capabilities.minDuration}
              max={capabilities.maxDuration}
              onChange={(duration) => onChange({ duration })}
            />
          )}
        </div>

        {mode === "video" && (
          <>
            <div>
              <label className="text-[10px] font-medium text-muted-foreground">
                随机种子
              </label>
              <Input
                type="number"
                value={form.seed}
                onChange={(event) => onChange({ seed: event.target.value })}
                placeholder="随机"
                className="mt-2 h-9 rounded-xl bg-background/65"
              />
            </div>
            <div className="grid grid-cols-3 gap-2">
              <ToggleOption
                label="生成声音"
                checked={form.generateAudio}
                disabled={!capabilities?.supportGenerateAudio}
                onChange={(generateAudio) => onChange({ generateAudio })}
              />
              <ToggleOption
                label="固定镜头"
                checked={form.cameraFixed}
                disabled={!capabilities?.supportCameraFixed}
                onChange={(cameraFixed) => onChange({ cameraFixed })}
              />
              <ToggleOption
                label="添加水印"
                checked={form.watermark}
                disabled={!capabilities?.supportWatermark}
                onChange={(watermark) => onChange({ watermark })}
              />
            </div>
          </>
        )}
      </div>
    </section>
  );
}
