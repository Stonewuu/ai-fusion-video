"use client";

import { useEffect, useId, useRef } from "react";
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
      className="fixed z-70 w-52 overflow-y-auto rounded-2xl border border-border/40 bg-popover/80 p-2 shadow-xl backdrop-blur-xl"
      style={{
        left: Math.max(8, Math.min(position.x, window.innerWidth - 216)),
        top: Math.max(
          8,
          Math.min(position.y, window.innerHeight - actions.length * 36 - 24),
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
      {actions.map((action) => (
        <CanvasSelectionActionButton
          key={action.id}
          action={action}
          menu
          owner={owner}
          onAction={onClose}
        />
      ))}
    </div>,
    document.body,
  );
}
