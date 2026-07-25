"use client";

import { useCallback, useEffect, useRef, type PointerEvent as ReactPointerEvent } from "react";
import { Bot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  ASSISTANT_LAUNCHER_SIZE,
  clampLauncherPosition,
  type AssistantPoint,
} from "./geometry";
import { useAssistantStore } from "@/lib/store/assistant-store";

interface AssistantLauncherProps {
  collapsed: boolean;
  onOpen: () => void;
  onPositionPaint: (position: AssistantPoint) => void;
}

export function AssistantLauncher({
  collapsed,
  onOpen,
  onPositionPaint,
}: AssistantLauncherProps) {
  const position = useAssistantStore((state) => state.launcherPosition);
  const updatePosition = useAssistantStore((state) => state.updateLauncherPosition);
  const pointerRef = useRef<{ x: number; y: number; originX: number; originY: number } | null>(null);
  const livePositionRef = useRef(position);
  const dragFrameRef = useRef<number | null>(null);
  const draggedRef = useRef(false);

  const paintPosition = useCallback((nextPosition: typeof position) => {
    onPositionPaint(nextPosition);
  }, [onPositionPaint]);

  const schedulePaint = useCallback(() => {
    if (dragFrameRef.current !== null) return;
    dragFrameRef.current = requestAnimationFrame(() => {
      dragFrameRef.current = null;
      paintPosition(livePositionRef.current);
    });
  }, [paintPosition]);

  useEffect(() => () => {
    if (dragFrameRef.current !== null) cancelAnimationFrame(dragFrameRef.current);
  }, []);

  const handlePointerDown = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (!collapsed) return;
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
    if (!collapsed) return;
    const pointer = pointerRef.current;
    if (!pointer) return;
    const deltaX = event.clientX - pointer.x;
    const deltaY = event.clientY - pointer.y;
    if (!draggedRef.current && Math.hypot(deltaX, deltaY) > 4) {
      draggedRef.current = true;
    }
    if (draggedRef.current) {
      livePositionRef.current = clampLauncherPosition({
        x: pointer.originX + deltaX,
        y: pointer.originY + deltaY,
      });
      schedulePaint();
    }
  };

  const finishPointer = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (!collapsed) return;
    if (pointerRef.current) {
      if (dragFrameRef.current !== null) {
        cancelAnimationFrame(dragFrameRef.current);
        dragFrameRef.current = null;
      }
      paintPosition(livePositionRef.current);
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
    <div
      aria-hidden={!collapsed}
      className="absolute left-0 top-0 z-20"
      style={{
        width: ASSISTANT_LAUNCHER_SIZE,
        height: ASSISTANT_LAUNCHER_SIZE,
        pointerEvents: collapsed ? "auto" : "none",
      }}
    >
      <Tooltip>
        <TooltipTrigger
          render={
          <Button
            type="button"
            variant="ghost"
            size="icon-lg"
            aria-label="打开融光助手"
            tabIndex={collapsed ? 0 : -1}
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
              if (collapsed) onOpen();
            }}
            style={{ touchAction: "none" }}
            className="relative size-full rounded-full text-primary"
          >
            <span className="flex items-center justify-center">
              <Bot className="size-7" />
            </span>
          </Button>
          }
        />
        <TooltipContent>融光助手</TooltipContent>
      </Tooltip>
    </div>
  );
}
