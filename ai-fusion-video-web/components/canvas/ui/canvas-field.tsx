"use client";

import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

/**
 * 画布输入控件：清透微底与细腻微边框，聚焦切换至纯背景、品牌色描边与柔和 3px 外圈。
 */
const canvasFieldClass =
  "w-full min-w-0 border border-input bg-foreground/[0.025] text-foreground shadow-[inset_0_1px_2px_rgba(0,0,0,0.02)] outline-none transition-[color,background-color,border-color,box-shadow] duration-150 placeholder:text-muted-foreground/65 hover:bg-foreground/[0.035] focus-visible:border-primary/85 focus-visible:bg-background focus-visible:ring-[3px] focus-visible:ring-primary/18 focus-visible:shadow-none disabled:cursor-not-allowed disabled:opacity-45 aria-invalid:border-destructive/75 aria-invalid:ring-[3px] aria-invalid:ring-destructive/18 dark:bg-white/[0.035] dark:hover:bg-white/[0.05] dark:focus-visible:border-primary dark:focus-visible:bg-background/95 dark:focus-visible:ring-primary/35 motion-reduce:transition-none";

export function CanvasInput({ className, ...props }: ComponentProps<"input">) {
  return (
    <input
      data-slot="canvas-input"
      className={cn(
        canvasFieldClass,
        "h-8 rounded-[10px] px-2.5 text-[13px]",
        className,
      )}
      {...props}
    />
  );
}

export function CanvasTextarea({
  className,
  ...props
}: ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="canvas-textarea"
      className={cn(
        canvasFieldClass,
        "field-sizing-fixed block min-h-16 resize-none rounded-[12px] px-3 py-2.5 text-sm leading-6",
        className,
      )}
      {...props}
    />
  );
}
