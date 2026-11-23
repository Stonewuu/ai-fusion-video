"use client";

import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

/** 画布四周的悬浮工具条容器：统一 40px 高度（含 32px 控件与 4px 内边距）、14px 圆角与多层柔影。 */
export function CanvasFloatingBar({
  className,
  ...props
}: ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "pointer-events-auto flex items-center gap-1 rounded-[14px] border border-foreground/[0.08] bg-popover/92 p-1 shadow-[0_1px_2px_rgba(0,0,0,0.04),0_8px_24px_-6px_rgba(0,0,0,0.1)] backdrop-blur-xl dark:border-white/[0.1] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_10px_28px_-6px_rgba(0,0,0,0.48)]",
        className,
      )}
      {...props}
    />
  );
}

/** 悬浮工具条内的分组分隔线。 */
export function CanvasBarDivider() {
  return (
    <span
      aria-hidden
      className="mx-0.5 h-4 w-px shrink-0 bg-foreground/[0.1] dark:bg-white/[0.12]"
    />
  );
}

/** 节点下方快捷面板（生成表单、正文面板）的脱离文档流浮层外壳。 */
export function CanvasPanel({ className, ...props }: ComponentProps<"section">) {
  return (
    <section
      className={cn(
        "flex min-h-0 flex-col overflow-hidden rounded-[20px] border border-foreground/[0.09] bg-popover/96 text-popover-foreground shadow-[0_2px_6px_rgba(0,0,0,0.04),0_18px_42px_-10px_rgba(0,0,0,0.14)] backdrop-blur-2xl dark:border-white/[0.11] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.09),0_22px_52px_-10px_rgba(0,0,0,0.62)]",
        className,
      )}
      {...props}
    />
  );
}
