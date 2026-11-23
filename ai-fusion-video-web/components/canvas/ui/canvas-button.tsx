"use client";

import { Button as ButtonPrimitive } from "@base-ui/react/button";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

/**
 * 画布专用按钮：用于节点、画布浮层（快捷面板、工具条、菜单、状态栏）。
 * 采用画布圆角体系（10px / 8px）与细腻微质感，适合叠加在画布背景与节点之上。
 * 普通页面、侧栏和 Dialog 仍使用 components/ui/button。
 */
export const canvasButtonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap font-medium outline-none select-none transition-[color,background-color,border-color,box-shadow,transform,filter] duration-150 focus-visible:ring-[3px] focus-visible:ring-primary/30 disabled:pointer-events-none disabled:opacity-35 motion-reduce:transition-none [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        /** 区域唯一主操作。 */
        primary:
          "bg-linear-to-b from-primary/92 to-primary text-primary-foreground shadow-[inset_0_1px_0_rgba(255,255,255,0.22),0_1px_2px_rgba(0,0,0,0.14)] hover:brightness-105 active:scale-[0.98]",
        /** 实体微边次操作，用于成组操作、参数芯片与素材标签。 */
        secondary:
          "border border-foreground/[0.09] bg-background/90 text-foreground shadow-[0_1px_2px_rgba(0,0,0,0.04)] hover:border-foreground/[0.16] hover:bg-background aria-expanded:border-primary/40 aria-expanded:bg-primary/[0.06] dark:border-white/[0.1] dark:bg-white/[0.06] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.06),0_1px_2px_rgba(0,0,0,0.22)] dark:hover:border-white/[0.16] dark:hover:bg-white/[0.1]",
        /** 工具条、菜单项与节点上的轻操作；aria-pressed 表示开启。 */
        ghost:
          "text-foreground/80 hover:bg-foreground/[0.06] hover:text-foreground aria-expanded:bg-foreground/[0.06] aria-expanded:text-foreground aria-pressed:bg-primary/12 aria-pressed:text-primary",
        /** 危险操作。 */
        danger:
          "text-destructive hover:bg-destructive/10 focus-visible:ring-destructive/30",
        /** 叠在图片、视频或点阵背景上的浮起按钮，自带实体底色与柔和投影。 */
        overlay:
          "border border-foreground/[0.1] bg-popover/95 text-foreground shadow-[0_2px_8px_-1px_rgba(0,0,0,0.12),0_1px_2px_rgba(0,0,0,0.06)] backdrop-blur-md hover:border-foreground/20 hover:bg-popover active:scale-[0.97] dark:border-white/[0.14] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_4px_12px_rgba(0,0,0,0.45)]",
        /** 生成与运行主操作：克制的反色单色质感（浅色墨曜黑、暗色冷白瓷），避免高饱和渐变喧宾夺主。 */
        generate:
          "bg-foreground text-background shadow-[inset_0_1px_0_rgba(255,255,255,0.16),0_1px_2px_rgba(0,0,0,0.18),0_4px_12px_-3px_rgba(0,0,0,0.2)] hover:bg-foreground/88 hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.2),0_2px_4px_rgba(0,0,0,0.2),0_6px_16px_-4px_rgba(0,0,0,0.24)] active:scale-[0.98] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.7),0_1px_2px_rgba(0,0,0,0.35),0_4px_14px_-3px_rgba(0,0,0,0.5)] dark:hover:bg-foreground/90",
      },
      size: {
        md: "h-8 rounded-[10px] px-3 text-[13px]",
        sm: "h-7 gap-1 rounded-[8px] px-2.5 text-xs [&_svg:not([class*='size-'])]:size-3.5",
        icon: "size-8 rounded-[10px]",
        "icon-sm": "size-7 rounded-[8px] [&_svg:not([class*='size-'])]:size-3.5",
        "icon-xs": "size-6 rounded-[8px] [&_svg:not([class*='size-'])]:size-3.5",
      },
      shape: {
        default: "",
        /** 仅用于节点外侧的圆形图标按钮（如添加后续）。 */
        round: "rounded-full",
      },
    },
    defaultVariants: {
      variant: "ghost",
      size: "md",
      shape: "default",
    },
  },
);

export type CanvasButtonProps = ButtonPrimitive.Props &
  VariantProps<typeof canvasButtonVariants>;

export function CanvasButton({
  className,
  variant,
  size,
  shape,
  ...props
}: CanvasButtonProps) {
  return (
    <ButtonPrimitive
      data-slot="canvas-button"
      className={cn(canvasButtonVariants({ variant, size, shape }), className)}
      {...props}
    />
  );
}
