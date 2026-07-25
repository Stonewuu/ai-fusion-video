"use client";

import { useRef, type PointerEvent as ReactPointerEvent } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Bot, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useAssistantStore } from "@/lib/store/assistant-store";

interface AssistantLauncherProps {
  canDock: boolean;
}

export function AssistantLauncher({ canDock }: AssistantLauncherProps) {
  const reducedMotion = useReducedMotion();
  const position = useAssistantStore((state) => state.launcherPosition);
  const openAssistant = useAssistantStore((state) => state.openAssistant);
  const hasRunning = useAssistantStore((state) =>
    Object.values(state.conversationStates).some((runtime) =>
      runtime.status === "running"
        || runtime.status === "pending"
        || runtime.status === "RUNNING"
        || runtime.status === "WAITING_CONFIRMATION"
        || runtime.status === "WAITING_EXTERNAL"
        || runtime.status === "CANCEL_REQUESTED",
    ),
  );
  const hasUnread = useAssistantStore((state) =>
    Object.values(state.conversationStates).some((runtime) => runtime.unread),
  );
  const updatePosition = useAssistantStore((state) => state.updateLauncherPosition);
  const pointerRef = useRef<{ x: number; y: number; originX: number; originY: number } | null>(null);
  const livePositionRef = useRef(position);
  const draggedRef = useRef(false);

  const handlePointerDown = (event: ReactPointerEvent<HTMLButtonElement>) => {
    livePositionRef.current = position;
    pointerRef.current = {
      x: event.clientX,
      y: event.clientY,
      originX: position.x,
      originY: position.y,
    };
    draggedRef.current = false;
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const pointer = pointerRef.current;
    if (!pointer) return;
    const deltaX = event.clientX - pointer.x;
    const deltaY = event.clientY - pointer.y;
    if (!draggedRef.current && Math.hypot(deltaX, deltaY) > 4) draggedRef.current = true;
    if (draggedRef.current) {
      livePositionRef.current = { x: pointer.originX + deltaX, y: pointer.originY + deltaY };
      updatePosition(livePositionRef.current);
    }
  };

  const finishPointer = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (pointerRef.current) {
      updatePosition(livePositionRef.current, true);
    }
    pointerRef.current = null;
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer capture may already have been released by the browser.
    }
  };

  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            type="button"
            variant="ai"
            size="icon-lg"
            aria-label="打开融光助手"
            title="打开融光助手"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={finishPointer}
            onPointerCancel={finishPointer}
            onClick={(event) => {
              if (draggedRef.current) {
                event.preventDefault();
                event.stopPropagation();
                draggedRef.current = false;
                return;
              }
              openAssistant(canDock);
            }}
            style={{
              left: position.x,
              top: position.y,
              position: "fixed",
              zIndex: 70,
              touchAction: "none",
            }}
            className="relative"
          >
            <motion.span
              layoutId="fusion-assistant-icon"
              transition={{ duration: reducedMotion ? 0.01 : 0.5, ease: [0.65, 0, 0.35, 1] }}
              className="flex items-center justify-center"
            >
              {hasRunning ? <Loader2 className="animate-spin motion-reduce:animate-none" /> : <Bot />}
            </motion.span>
            {hasUnread ? (
              <span
                aria-label="有未读回复"
                className="absolute right-1 top-1 size-2.5 rounded-full bg-primary ring-2 ring-background"
              />
            ) : null}
            {hasRunning ? (
              <span className="absolute -inset-1 -z-10 rounded-2xl border border-primary/20 motion-safe:animate-pulse" />
            ) : null}
          </Button>
        }
      />
      <TooltipContent>融光助手</TooltipContent>
    </Tooltip>
  );
}
