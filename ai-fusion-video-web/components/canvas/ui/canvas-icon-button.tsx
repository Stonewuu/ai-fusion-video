"use client";

import type { ReactNode } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CanvasButton, type CanvasButtonProps } from "./canvas-button";

/**
 * 画布图标按钮：必须提供 label，同时作为无障碍名称与悬停提示；
 * shortcut 会显示在提示中。弹层触发器等需要 render 组合的场景请直接使用 CanvasButton 并设置 aria-label。
 */
export function CanvasIconButton({
  label,
  hint,
  shortcut,
  size = "icon",
  side = "top",
  sideOffset,
  children,
  ...props
}: Omit<CanvasButtonProps, "children" | "size"> & {
  label: string;
  /** 补充说明，默认与 label 相同。 */
  hint?: string;
  shortcut?: string;
  size?: "icon" | "icon-sm" | "icon-xs";
  side?: "top" | "bottom" | "left" | "right";
  sideOffset?: number;
  children: ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={<CanvasButton size={size} aria-label={label} {...props} />}
      >
        {children}
      </TooltipTrigger>
      <TooltipContent side={side} sideOffset={sideOffset}>
        {hint ?? label}
        {shortcut && (
          <kbd className="rounded-sm bg-background/20 px-1 font-sans text-xs">
            {shortcut}
          </kbd>
        )}
      </TooltipContent>
    </Tooltip>
  );
}
