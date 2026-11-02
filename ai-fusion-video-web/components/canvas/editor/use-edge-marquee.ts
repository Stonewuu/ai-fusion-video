"use client";

import { useRef, type PointerEvent as ReactPointerEvent } from "react";

const EDGE_MARQUEE_DISTANCE = 4;

function pointerEventInit(event: ReactPointerEvent<HTMLDivElement>) {
  return {
    bubbles: true,
    cancelable: true,
    composed: true,
    pointerId: event.pointerId,
    pointerType: event.pointerType,
    isPrimary: event.isPrimary,
    button: event.button,
    buttons: event.buttons,
    clientX: event.clientX,
    clientY: event.clientY,
    ctrlKey: event.ctrlKey,
    metaKey: event.metaKey,
    shiftKey: event.shiftKey,
    altKey: event.altKey,
  };
}

interface EdgeMarqueeStart {
  pane: HTMLDivElement;
  pointer: ReturnType<typeof pointerEventInit>;
}

export function useCanvasEdgeMarquee(enabled: boolean) {
  const pending = useRef<EdgeMarqueeStart | null>(null);

  const onPointerDownCapture = (event: ReactPointerEvent<HTMLDivElement>) => {
    pending.current = null;
    if (
      !enabled ||
      event.button !== 0 ||
      !event.isPrimary ||
      event.pointerType === "touch" ||
      event.shiftKey ||
      !(event.target instanceof Element)
    ) {
      return;
    }
    const edge = event.target.closest(".react-flow__edge");
    if (!edge || event.target.closest(".react-flow__edgeupdater")) {
      return;
    }
    pending.current = {
      pane: edge.closest<HTMLDivElement>(".react-flow__pane")!,
      pointer: pointerEventInit(event),
    };
  };

  const onPointerMoveCapture = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = pending.current;
    if (!start || event.pointerId !== start.pointer.pointerId) {
      return;
    }
    if (!enabled || (event.buttons & 1) === 0) {
      pending.current = null;
      return;
    }
    const distance = Math.hypot(
      event.clientX - start.pointer.clientX,
      event.clientY - start.pointer.clientY,
    );
    if (distance < EDGE_MARQUEE_DISTANCE) {
      return;
    }

    // 确认拖动后才转交给画布，保留单击连线的行为，并沿用画布已有的框选与自动平移处理。
    pending.current = null;
    event.preventDefault();
    event.stopPropagation();
    start.pane.dispatchEvent(new PointerEvent("pointerdown", start.pointer));
    start.pane.dispatchEvent(new PointerEvent("pointermove", pointerEventInit(event)));
  };

  const clearPending = () => {
    pending.current = null;
  };

  return {
    onPointerDownCapture,
    onPointerMoveCapture,
    onPointerUpCapture: clearPending,
    onPointerCancelCapture: clearPending,
    onPointerLeave: clearPending,
  };
}
