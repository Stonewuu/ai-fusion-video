"use client";

import { Fragment, useEffect, useId, useRef } from "react";
import { createPortal } from "react-dom";
import { CanvasSelectionActionButton } from "./selection-action-button";
import type { CanvasSelectionAction } from "./build-context-actions";

export type CanvasContextAction = CanvasSelectionAction;
export function CanvasContextActions({
  position,
  actions,
  onClose,
  label = "节点操作",
}: {
  position: { x: number; y: number };
  actions: CanvasContextAction[];
  onClose: () => void;
  label?: string;
}) {
  const root = useRef<HTMLDivElement>(null);
  const owner = useId();
  useEffect(() => {
    const previous = document.activeElement;
    root.current
      ?.querySelector<HTMLButtonElement>("button:not(:disabled)")
      ?.focus();
    const outside = (event: PointerEvent) => {
      if (
        event.target instanceof Element &&
        event.target.closest(`[data-canvas-menu-owner="${owner}"]`)
      )
        return;
      if (event.target instanceof Node && !root.current?.contains(event.target))
        onClose();
    };
    document.addEventListener("pointerdown", outside);
    return () => {
      document.removeEventListener("pointerdown", outside);
      if (previous instanceof HTMLElement && previous.isConnected)
        previous.focus();
    };
  }, [onClose, owner]);
  return createPortal(
    <div
      ref={root}
      role="menu"
      aria-label={label}
      className="fixed z-70 flex w-52 flex-col gap-0.5 overflow-y-auto rounded-[14px] border border-foreground/[0.09] bg-popover/95 p-1 shadow-[0_4px_14px_rgba(0,0,0,0.08),0_18px_40px_-10px_rgba(0,0,0,0.16)] backdrop-blur-2xl dark:border-white/[0.12] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_18px_44px_rgba(0,0,0,0.58)]"
      style={{
        left: Math.max(8, Math.min(position.x, window.innerWidth - 216)),
        top: Math.max(
          8,
          Math.min(position.y, window.innerHeight - actions.length * 34 - 16),
        ),
        maxHeight: window.innerHeight - 16,
      }}
      onContextMenu={(event) => event.preventDefault()}
      onKeyDown={(event) => {
        if (!event.currentTarget.contains(event.target as Node)) return;
        if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          onClose();
        }
        if (event.key === "Tab") onClose();
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          const buttons = [
            ...root.current!.querySelectorAll<HTMLButtonElement>(
              "button:not(:disabled)",
            ),
          ];
          const index = buttons.findIndex(
            (button) => button === document.activeElement,
          );
          buttons[
            (index + (event.key === "ArrowDown" ? 1 : buttons.length - 1)) %
              buttons.length
          ]?.focus();
        }
        if (event.key === "Home" || event.key === "End") {
          event.preventDefault();
          const buttons = root.current?.querySelectorAll<HTMLButtonElement>(
            "button:not(:disabled)",
          );
          buttons?.[event.key === "Home" ? 0 : buttons.length - 1]?.focus();
        }
      }}
    >
      {actions.map((action, index) => (
        <Fragment key={action.id}>
          {/* 危险操作与普通操作之间加分隔，降低误触。 */}
          {action.destructive && index > 0 && (
            <div role="separator" className="mx-2 my-0.5 h-px bg-border/40" />
          )}
          <CanvasSelectionActionButton
            action={action}
            menu
            owner={owner}
            onAction={onClose}
          />
        </Fragment>
      ))}
    </div>,
    document.body,
  );
}
