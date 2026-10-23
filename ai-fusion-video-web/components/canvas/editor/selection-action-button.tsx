"use client";

import { useRef, useState } from "react";
import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { CanvasSelectionAction } from "./build-context-actions";

export function CanvasSelectionActionButton({
  action,
  menu = false,
  owner,
  onAction,
}: {
  action: CanvasSelectionAction;
  menu?: boolean;
  owner?: string;
  onAction?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const handoffFocus = useRef(false);
  const Icon = action.icon;
  if (action.children)
    return (
      <Popover
        open={open}
        onOpenChange={(next) => {
          if (next) handoffFocus.current = false;
          setOpen(next);
        }}
      >
        <PopoverTrigger
          onKeyDown={(event) => {
            if (menu && event.key === "ArrowRight") {
              event.preventDefault();
              setOpen(true);
            }
          }}
          render={
            <Button
              variant="ghost"
              size="default"
              disabled={action.disabled}
              role={menu ? "menuitem" : undefined}
              className={menu ? "w-full justify-start" : undefined}
            />
          }
        >
          <Icon />
          {action.label}
          {menu && <ChevronRight className="ml-auto" />}
        </PopoverTrigger>
        <PopoverContent
          finalFocus={() => !handoffFocus.current}
          align={menu ? "start" : "center"}
          side={menu ? "right" : "bottom"}
          role="menu"
          className="w-52 max-w-[calc(100vw-2rem)] gap-0 border border-border/40 bg-popover/80 p-2 shadow-xl backdrop-blur-xl"
          data-canvas-menu-owner={owner}
          onKeyDown={(event) => {
            if (["ArrowLeft", "Escape"].includes(event.key)) {
              event.preventDefault();
              event.stopPropagation();
              setOpen(false);
              return;
            }
            if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key))
              return;
            event.preventDefault();
            event.stopPropagation();
            const buttons = [
              ...event.currentTarget.querySelectorAll<HTMLButtonElement>(
                "button:not(:disabled)",
              ),
            ];
            const index = buttons.findIndex(
              (button) => button === document.activeElement,
            );
            const next =
              event.key === "Home"
                ? 0
                : event.key === "End"
                  ? buttons.length - 1
                  : (index +
                      (event.key === "ArrowDown" ? 1 : buttons.length - 1)) %
                    buttons.length;
            buttons[next]?.focus();
          }}
        >
          <PopoverTitle className="sr-only">{action.label}</PopoverTitle>
          {action.children.map((child) => (
            <Button
              key={child.id}
              role="menuitem"
              variant="ghost"
              size="default"
              className="w-full justify-start"
              disabled={action.disabled || child.disabled}
              onClick={(event) => {
                handoffFocus.current = !!child.opensOverlay;
                child.action?.(event.currentTarget);
                setOpen(false);
                onAction?.();
              }}
            >
              <child.icon />
              {child.label}
            </Button>
          ))}
        </PopoverContent>
      </Popover>
    );
  return (
    <Button
      role={menu ? "menuitem" : undefined}
      variant={action.destructive ? "destructive" : "ghost"}
      size="default"
      className={menu ? "w-full justify-start" : undefined}
      disabled={action.disabled}
      title={
        action.id === "remove"
          ? "从画布移除，保留项目中的项目内容"
          : action.id === "copy"
            ? "复制为新的画布引用，项目内容保持关联"
            : undefined
      }
      onClick={(event) => {
        action.action?.(event.currentTarget);
        onAction?.();
      }}
    >
      <Icon />
      {action.label}
    </Button>
  );
}
