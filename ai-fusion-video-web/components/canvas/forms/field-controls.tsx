"use client";

import { useId, useState } from "react";
import { Minus, Plus, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { CanvasSegmentedControl } from "../workspace/segmented-control";
import { CanvasSelect } from "./canvas-select";
import type { FieldChoice } from "@/lib/canvas/field-presentation";

export function CanvasChoiceField({ label, value, items, onChange, disabled = false, custom = false, placeholder = "请选择", invalid = false }: {
  label: string; value?: string | null; items: FieldChoice[]; onChange: (value: string) => void;
  disabled?: boolean; custom?: boolean; placeholder?: string; invalid?: boolean;
}) {
  const [editingCustom, setEditingCustom] = useState(false);
  const known = items.some((item) => item.value === value);
  const showCustom = custom && (editingCustom || value != null && !known);
  const options = [...items];
  if (value != null && !known && !custom) options.push({ value, label: `当前：${value}` });
  if (custom) options.push({ value: "__custom__", label: "自定义" });
  const selected = showCustom ? "__custom__" : value ?? "";
  const select = (next: string) => {
    if (custom && next === "__custom__") { setEditingCustom(true); return; }
    setEditingCustom(false);
    onChange(next);
  };
  return <div className="flex min-w-0 flex-col gap-2">
    {options.length <= 3 && options.every((item) => item.label.length <= 8) ? (
      <CanvasSegmentedControl label={label} value={selected} className="w-full"
        options={options.map((item) => ({ ...item, disabled }))} onChange={select} />
    ) : <CanvasSelect label={label} value={value == null && !showCustom ? null : selected} items={options}
      disabled={disabled} placeholder={placeholder} onChange={select} />}
    {showCustom && <Input aria-label={`自定义${label}`} aria-invalid={invalid || undefined} value={value ?? ""}
      placeholder={`填写${label}`} disabled={disabled} onChange={(event) => onChange(event.target.value)} />}
  </div>;
}

export function CanvasBooleanField({ label, value, onChange, disabled, unsupported = false, defaultLabel = "默认", defaultTitle = "不指定，使用模型或工作流的默认设置" }: {
  label: string; value: boolean | undefined; onChange: (next: boolean | undefined) => void; disabled?: boolean; unsupported?: boolean;
  defaultLabel?: string; defaultTitle?: string;
}) {
  return <div className="flex flex-col gap-2">
    <CanvasSegmentedControl label={label} value={value === undefined ? "default" : value ? "on" : "off"} className="w-full"
      options={[
        { value: "default", label: defaultLabel, title: defaultTitle, disabled },
        { value: "on", label: "开启", disabled: disabled || unsupported },
        { value: "off", label: "关闭", disabled },
      ]} onChange={(next) => onChange(next === "default" ? undefined : next === "on")} />
    {unsupported && <p className="text-xs text-muted-foreground">当前模型不支持开启此选项。</p>}
  </div>;
}

export function CanvasNumberField({ id, label, value, onChange, min, max, step = 1, disabled = false, presets = [], unit, placeholder = "默认", range = false, invalid = false }: {
  id?: string; label: string; value: number | undefined; onChange: (value: number | undefined) => void;
  min?: number; max?: number; step?: number; disabled?: boolean; presets?: number[]; unit?: string;
  placeholder?: string; range?: boolean; invalid?: boolean;
}) {
  const generatedId = useId();
  const current = value ?? min ?? 0;
  const changeBy = (direction: number) => {
    if (value === undefined) { onChange(min ?? (direction > 0 ? step : 0)); return; }
    const next = Number((current + direction * step).toFixed(6));
    onChange(Math.min(max ?? Infinity, Math.max(min ?? -Infinity, next)));
  };
  return <div className="flex min-w-0 flex-col gap-2">
    <div className="flex min-w-0 items-center gap-2">
      <Button variant="outline" size="icon" disabled={disabled || min !== undefined && current <= min} aria-label={`减少${label}`} onClick={() => changeBy(-1)}><Minus /></Button>
      <div className="relative min-w-0 flex-1">
        <Input id={id ?? generatedId} type="number" aria-label={label} aria-invalid={invalid || undefined}
          value={value ?? ""} min={min} max={max} step={step} disabled={disabled} placeholder={placeholder}
          className={unit ? "pr-10" : undefined}
          onChange={(event) => onChange(event.target.value === "" ? undefined : event.target.valueAsNumber)} />
        {unit && <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-xs text-muted-foreground">{unit}</span>}
      </div>
      <Button variant="outline" size="icon" disabled={disabled || max !== undefined && current >= max} aria-label={`增加${label}`} onClick={() => changeBy(1)}><Plus /></Button>
    </div>
    {range && min !== undefined && max !== undefined && max > min && value !== undefined && (
      <input type="range" aria-label={`${label}滑块`} aria-valuetext={`${value}${unit ?? ""}`} min={min} max={max} step={step}
        value={value} disabled={disabled} onChange={(event) => onChange(Number(event.target.value))}
        className="h-9 w-full cursor-pointer rounded-lg accent-primary outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-default disabled:opacity-50" />
    )}
    {!!presets.length && <div className="flex flex-wrap gap-2" role="group" aria-label={`${label}常用值`}>
      {presets.filter((item) => (min === undefined || item >= min) && (max === undefined || item <= max)).map((item) => (
        <Button key={item} variant={value === item ? "secondary" : "outline"} size="sm" aria-pressed={value === item} disabled={disabled} onClick={() => onChange(item)}>{item}{unit}</Button>
      ))}
    </div>}
  </div>;
}

export function CanvasFieldReset({ label, disabled, onReset }: { label: string; disabled?: boolean; onReset: () => void }) {
  return <Button variant="ghost" size="icon-xs" aria-label={`重置${label}`} title="恢复默认设置" disabled={disabled} onClick={onReset}><RotateCcw /></Button>;
}
