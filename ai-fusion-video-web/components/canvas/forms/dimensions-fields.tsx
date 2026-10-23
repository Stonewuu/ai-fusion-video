"use client";

import { useState } from "react";
import { ArrowLeftRight, LockKeyhole, LockKeyholeOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { CanvasCapability, CanvasConfig } from "@/lib/canvas/types";
import { CanvasNumberField } from "./field-controls";

export function CanvasDimensionsFields({ capability, config, onChange, disabled }: {
  capability: CanvasCapability; config: CanvasConfig; onChange: (next: CanvasConfig) => void; disabled: boolean;
}) {
  const [lockedRatio, setLockedRatio] = useState<number | null>(null);
  const width = typeof config.width === "number" ? config.width : undefined;
  const height = typeof config.height === "number" ? config.height : undefined;
  const set = (field: "width" | "height", value: number | undefined) => {
    const next = { ...config };
    if (value === undefined) delete next[field]; else next[field] = value;
    if (lockedRatio && value !== undefined) next[field === "width" ? "height" : "width"] = Math.max(1, Math.round(field === "width" ? value / lockedRatio : value * lockedRatio));
    onChange(next);
  };
  const body = <div className="flex min-w-0 flex-col gap-3">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <span className="text-sm font-medium">{capability.kind === "imageTransform" && config.operation === "crop" ? "裁剪区域尺寸" : "宽高尺寸"}</span>
      <div className="flex gap-2">
        <Button variant="ghost" size="icon" title="交换宽高" aria-label="交换宽高" disabled={disabled || width === undefined || height === undefined}
          onClick={() => { onChange({ ...config, width: height!, height: width! }); if (lockedRatio) setLockedRatio(1 / lockedRatio); }}><ArrowLeftRight /></Button>
        <Button variant={lockedRatio ? "secondary" : "ghost"} size="icon" title={lockedRatio ? "解锁比例" : "锁定当前宽高比例"}
          aria-label="锁定宽高比例" aria-pressed={!!lockedRatio} disabled={disabled || !width || !height}
          onClick={() => setLockedRatio(lockedRatio ? null : width! / height!)}>{lockedRatio ? <LockKeyhole /> : <LockKeyholeOpen />}</Button>
      </div>
    </div>
    {(["width", "height"] as const).map((field) => <div key={field} className="flex flex-col gap-2">
      <span className="text-xs text-muted-foreground">{field === "width" ? "宽度" : "高度"}</span>
      <CanvasNumberField label={field === "width" ? "宽度" : "高度"} value={field === "width" ? width : height}
        min={capability.configFields[field].minimum ?? 1} max={capability.configFields[field].maximum ?? undefined}
        unit="px" disabled={disabled} onChange={(value) => set(field, value)} />
      {typeof config[field] === "number" && (config[field] < (capability.configFields[field].minimum ?? 1) || config[field] > (capability.configFields[field].maximum ?? Infinity)) &&
        <p role="alert" className="text-xs text-destructive">尺寸超出允许范围，请调整后保存。</p>}
    </div>)}
    <div className="flex flex-wrap gap-2" role="group" aria-label="常用画面尺寸">
      {[[1024, 1024, "方形"], [1920, 1080, "横屏"], [1080, 1920, "竖屏"]].map(([w, h, label]) => (
        <Button key={label} variant={width === w && height === h ? "secondary" : "outline"} size="sm" disabled={disabled}
          title={`${w} × ${h} px`} onClick={() => { onChange({ ...config, width: Number(w), height: Number(h) }); if (lockedRatio) setLockedRatio(Number(w) / Number(h)); }}>{label}</Button>
      ))}
    </div>
    {capability.kind === "imageGeneration" && (width !== undefined || height !== undefined) && <Button variant="ghost" disabled={disabled} onClick={() => {
      const next = { ...config }; delete next.width; delete next.height; setLockedRatio(null); onChange(next);
    }}>恢复模型默认尺寸</Button>}
    {capability.kind === "imageTransform" && config.operation === "crop" && <p className="text-xs text-muted-foreground">裁剪范围需位于原图内；下方可以调整起点。</p>}
  </div>;
  return capability.kind === "imageGeneration" ? <details className="min-w-0">
    <summary className="cursor-pointer rounded-lg text-sm text-muted-foreground outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">自定义像素尺寸{width || height ? " · 已设置" : ""}</summary>
    <div className="pt-3">{body}</div>
  </details> : body;
}
