"use client";

import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Loader2 } from "lucide-react";
import { AssistantComposer } from "./composer";
import { ConversationNavigation, AssistantEmptyState } from "./conversation-navigation";
import { AssistantMessageList } from "./message-list";
import { AssistantTitleBar } from "./title-bar";
import {
  clampRect,
  resizeRect,
  type AssistantRect,
  type ResizeDirection,
} from "./geometry";
import { useAssistantStore, type AssistantMode } from "@/lib/store/assistant-store";

interface AssistantWindowProps {
  mode: AssistantMode;
  canDock: boolean;
  mobileViewport: boolean;
  showLeadingIcon: boolean;
  projectId?: number | null;
  onToggleDock: () => void;
  onToggleMaximize: () => void;
  onClose: () => void;
  onRectPaint: (rect: AssistantRect) => void;
  interactionGuardActive: boolean;
  onInteractionGuardRelease: () => void;
  contentActive: boolean;
  onDockResizeStart?: (event: ReactPointerEvent<HTMLDivElement>) => void;
}

const RESIZE_HANDLES: Array<{ direction: ResizeDirection; className: string; label: string }> = [
  { direction: "n", className: "inset-x-4 top-0 h-1 cursor-ns-resize", label: "调整窗口上边缘" },
  { direction: "s", className: "inset-x-4 bottom-0 h-1 cursor-ns-resize", label: "调整窗口下边缘" },
  { direction: "e", className: "inset-y-4 right-0 w-1 cursor-ew-resize", label: "调整窗口右边缘" },
  { direction: "w", className: "inset-y-4 left-0 w-1 cursor-ew-resize", label: "调整窗口左边缘" },
  { direction: "ne", className: "right-0 top-0 size-4 cursor-nesw-resize", label: "调整窗口右上角" },
  { direction: "nw", className: "left-0 top-0 size-4 cursor-nwse-resize", label: "调整窗口左上角" },
  { direction: "se", className: "bottom-0 right-0 size-4 cursor-nwse-resize", label: "调整窗口右下角" },
  { direction: "sw", className: "bottom-0 left-0 size-4 cursor-nesw-resize", label: "调整窗口左下角" },
];

function isInteractiveTarget(target: EventTarget | null) {
  return target instanceof HTMLElement && !!target.closest("[data-assistant-interactive],button,input,textarea,select,[role=combobox]");
}

export function AssistantWindow({
  mode,
  canDock,
  mobileViewport,
  showLeadingIcon,
  projectId,
  onToggleDock,
  onToggleMaximize,
  onClose,
  onRectPaint,
  interactionGuardActive,
  onInteractionGuardRelease,
  contentActive,
  onDockResizeStart,
}: AssistantWindowProps) {
  const selectedConversationId = useAssistantStore((state) => state.selectedConversationId);
  const updateNormalRect = useAssistantStore((state) => state.updateNormalRect);
  const setMode = useAssistantStore((state) => state.setMode);
  const setDrawerOpen = useAssistantStore((state) => state.setDrawerOpen);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ pointerX: number; pointerY: number; rect: AssistantRect } | null>(null);
  const resizeRef = useRef<{ direction: ResizeDirection; pointerX: number; pointerY: number; rect: AssistantRect } | null>(null);
  const liveRectRef = useRef<AssistantRect | null>(null);
  const interactionFrameRef = useRef<number | null>(null);
  const [inputHeight, setInputHeight] = useState(0);
  const [interactionActive, setInteractionActive] = useState(false);

  const paintRect = useCallback((rect: AssistantRect) => {
    onRectPaint(rect);
  }, [onRectPaint]);

  const scheduleRectPaint = useCallback(() => {
    if (interactionFrameRef.current !== null) return;
    interactionFrameRef.current = requestAnimationFrame(() => {
      interactionFrameRef.current = null;
      if (liveRectRef.current) paintRect(liveRectRef.current);
    });
  }, [paintRect]);

  useEffect(() => {
    document.body.style.userSelect = interactionActive ? "none" : "";
    document.body.style.cursor = interactionActive ? "grabbing" : "";
    return () => {
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
  }, [interactionActive]);

  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) return;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width ?? 0;
      if (width >= 650) setDrawerOpen(false);
    });
    observer.observe(shell);
    return () => observer.disconnect();
  }, [setDrawerOpen]);

  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (drag) {
        liveRectRef.current = clampRect({
          ...drag.rect,
          x: drag.rect.x + event.clientX - drag.pointerX,
          y: drag.rect.y + event.clientY - drag.pointerY,
        });
        scheduleRectPaint();
      }
      const resize = resizeRef.current;
      if (resize) {
        liveRectRef.current = resizeRect(
          resize.rect,
          resize.direction,
          event.clientX - resize.pointerX,
          event.clientY - resize.pointerY,
        );
        scheduleRectPaint();
      }
    };
    const onPointerUp = () => {
      if (dragRef.current || resizeRef.current) {
        if (interactionFrameRef.current !== null) {
          cancelAnimationFrame(interactionFrameRef.current);
          interactionFrameRef.current = null;
        }
        const nextRect = liveRectRef.current ?? useAssistantStore.getState().normalRect;
        paintRect(nextRect);
        updateNormalRect(nextRect, true);
      }
      dragRef.current = null;
      resizeRef.current = null;
      liveRectRef.current = null;
      setInteractionActive(false);
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
      if (interactionFrameRef.current !== null) {
        cancelAnimationFrame(interactionFrameRef.current);
        interactionFrameRef.current = null;
      }
    };
  }, [paintRect, scheduleRectPaint, updateNormalRect]);

  const startDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (mode === "docked" || isInteractiveTarget(event.target)) return;
    if (mode === "maximized") {
      setMode("floating", canDock);
    }
    dragRef.current = {
      pointerX: event.clientX,
      pointerY: event.clientY,
      rect: useAssistantStore.getState().normalRect,
    };
    liveRectRef.current = dragRef.current.rect;
    setInteractionActive(true);
  };

  const startResize = (direction: ResizeDirection, event: ReactPointerEvent<HTMLDivElement>) => {
    if (mode !== "floating") return;
    event.stopPropagation();
    resizeRef.current = {
      direction,
      pointerX: event.clientX,
      pointerY: event.clientY,
      rect: useAssistantStore.getState().normalRect,
    };
    liveRectRef.current = resizeRef.current.rect;
    setInteractionActive(true);
  };

  return (
    <div ref={shellRef} className="assistant-shell relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden">
      <AssistantTitleBar
        mode={mode}
        canDock={canDock}
        mobileViewport={mobileViewport}
        showLeadingIcon={showLeadingIcon}
        onToggleDock={onToggleDock}
        onToggleMaximize={onToggleMaximize}
        onClose={onClose}
        onStartDrag={startDrag}
      />

      <div className="relative flex min-h-0 flex-1">
        {mode !== "collapsed" ? <ConversationNavigation /> : null}
        <section className="assistant-content relative flex min-w-0 flex-1 flex-col bg-background/20">
          {mode !== "collapsed"
            ? selectedConversationId
              ? contentActive ? (
                  <AssistantMessageList
                    key={selectedConversationId}
                    conversationId={selectedConversationId}
                    inputHeight={inputHeight}
                  />
                ) : (
                  <div className="flex min-h-0 flex-1 items-center justify-center gap-2 text-xs text-muted-foreground">
                    <Loader2 className="size-4 animate-spin motion-reduce:animate-none" /> 加载消息
                  </div>
                )
              : <AssistantEmptyState inputRef={inputRef} bottomInset={inputHeight} />
            : null}
          {mode !== "collapsed" ? (
            <AssistantComposer
              active={contentActive}
              tooltipsEnabled={!interactionGuardActive}
              projectId={projectId}
              inputRef={inputRef}
              onHeightChange={setInputHeight}
            />
          ) : null}
        </section>
      </div>

      {mode === "docked" && onDockResizeStart ? (
        <div
          role="separator"
          aria-label="调整助手宽度"
          tabIndex={0}
          className="absolute inset-y-0 left-0 z-50 w-1 touch-none cursor-ew-resize bg-border/40 transition-colors hover:bg-primary/60 focus-visible:bg-primary/60"
          data-assistant-interactive="true"
          onPointerDown={onDockResizeStart}
        />
      ) : null}

      {mode === "floating" ? RESIZE_HANDLES.map((handle) => (
        <div
          key={handle.direction}
          role="separator"
          aria-label={handle.label}
          className={`absolute z-20 touch-none ${handle.className}`}
          data-assistant-interactive="true"
          onPointerDown={(event) => startResize(handle.direction, event)}
        />
      )) : null}

      {interactionGuardActive ? (
        <div
          aria-hidden="true"
          className="absolute inset-0 z-60 cursor-default"
          onPointerMove={onInteractionGuardRelease}
          onPointerDown={onInteractionGuardRelease}
        />
      ) : null}
    </div>
  );
}
