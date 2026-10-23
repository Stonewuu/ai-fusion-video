import type { KeyboardEvent, MouseEvent, TouchEvent } from "react";
import { isEditableTarget } from "@/lib/canvas/layout-actions";

/** Keep existing drag listeners mounted while a command is being confirmed. */
export function createCanvasGestureGate(isPending: () => boolean) {
  const graphTarget = (target: EventTarget | null) =>
    target instanceof Element &&
    !!target.closest(
      ".react-flow__handle",
    ) &&
    !isEditableTarget(target) &&
    !target.closest("button,a");
  const block = (event: {
    preventDefault: () => void;
    stopPropagation: () => void;
  }) => {
    event.preventDefault();
    event.stopPropagation();
  };
  return {
    mouseDown: (event: MouseEvent<HTMLDivElement>) => {
      if (event.button === 0 && isPending() && graphTarget(event.target))
        block(event);
    },
    touchStart: (event: TouchEvent<HTMLDivElement>) => {
      if (
        event.touches.length === 1 &&
        isPending() &&
        graphTarget(event.target)
      )
        block(event);
    },
    click: (event: MouseEvent<HTMLDivElement>) => {
      if (
        isPending() &&
        event.target instanceof Element &&
        event.target.closest(".react-flow__handle")
      )
        block(event);
    },
    keyDown: (event: KeyboardEvent<HTMLDivElement>) => {
      if (
        !isPending() ||
        !graphTarget(event.target) ||
        !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)
      )
        return false;
      block(event);
      return true;
    },
  };
}
