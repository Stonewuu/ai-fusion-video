"use client";

import {
  type ReactNode,
  type RefObject,
  useEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const COLLAPSED_RESULT_HEIGHT = 256;
const FLOATING_BUTTON_HEIGHT = 36;
const FLOATING_BUTTON_INSET = 8;
const FLOATING_BUTTON_GAP = 8;

interface FloatingPlacement {
  left: number;
  top: number;
}

interface VisibleViewportRect {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

function scrollViewportRect(element: HTMLElement): VisibleViewportRect {
  let parent = element.parentElement;
  while (parent) {
    const overflowY = window.getComputedStyle(parent).overflowY;
    if (overflowY === "auto" || overflowY === "scroll") {
      const rect = parent.getBoundingClientRect();
      let bottom = rect.bottom;
      if (parent.classList.contains("assistant-message-viewport")) {
        const assistantComposerInset = Number.parseFloat(
          window.getComputedStyle(parent).paddingBottom,
        );
        if (!Number.isFinite(assistantComposerInset)) {
          throw new Error("Assistant message viewport has an invalid composer inset");
        }
        bottom -= assistantComposerInset;
      }
      return {
        left: rect.left,
        right: rect.right,
        top: rect.top,
        bottom,
      };
    }
    parent = parent.parentElement;
  }
  throw new Error("Expanded tool result is not inside a scroll viewport");
}

function FloatingCollapseButton({
  resultRef,
  onCollapse,
}: {
  resultRef: RefObject<HTMLDivElement | null>;
  onCollapse: () => void;
}) {
  const [placement, setPlacement] = useState<FloatingPlacement | null>(null);

  useEffect(() => {
    const result = resultRef.current;
    if (!result) throw new Error("Expanded tool result has no container");

    let frameId: number | undefined;
    const update = () => {
      frameId = undefined;
      const rect = result.getBoundingClientRect();
      const scrollRect = scrollViewportRect(result);
      const viewportLeft = Math.max(scrollRect.left, 0);
      const viewportRight = Math.min(scrollRect.right, window.innerWidth);
      const viewportTop = Math.max(scrollRect.top, 0);
      const viewportBottom = Math.min(scrollRect.bottom, window.innerHeight);
      const visibleTop = Math.max(rect.top, viewportTop);
      const visibleBottom = Math.min(rect.bottom, viewportBottom);

      if (visibleBottom - visibleTop < 56 || viewportRight <= viewportLeft) {
        setPlacement((current) => current === null ? current : null);
        return;
      }

      const left = Math.round(Math.min(
        Math.max(rect.left + rect.width / 2, viewportLeft + 56),
        viewportRight - 56,
      ));
      let top = visibleBottom - FLOATING_BUTTON_HEIGHT - FLOATING_BUTTON_INSET;
      const messageList = result.closest<HTMLElement>("[data-assistant-message-list]");
      const backToBottom = messageList?.querySelector<HTMLElement>(
        "[data-assistant-back-to-bottom]",
      );
      if (backToBottom) {
        top = Math.min(
          top,
          backToBottom.getBoundingClientRect().top
            - FLOATING_BUTTON_HEIGHT
            - FLOATING_BUTTON_GAP,
        );
      }
      if (top < visibleTop + FLOATING_BUTTON_INSET) {
        setPlacement((current) => current === null ? current : null);
        return;
      }

      const next = {
        left,
        top: Math.round(top),
      };
      setPlacement((current) =>
        current?.left === next.left && current.top === next.top ? current : next
      );
    };
    const scheduleUpdate = () => {
      if (frameId !== undefined) return;
      frameId = requestAnimationFrame(update);
    };

    const resizeObserver = new ResizeObserver(scheduleUpdate);
    resizeObserver.observe(result);
    const messageList = result.closest<HTMLElement>("[data-assistant-message-list]");
    let mutationObserver: MutationObserver | undefined;
    if (messageList) {
      mutationObserver = new MutationObserver(scheduleUpdate);
      mutationObserver.observe(messageList, { childList: true, subtree: true });
    }
    document.addEventListener("scroll", scheduleUpdate, true);
    window.addEventListener("resize", scheduleUpdate);
    scheduleUpdate();

    return () => {
      if (frameId !== undefined) cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      mutationObserver?.disconnect();
      document.removeEventListener("scroll", scheduleUpdate, true);
      window.removeEventListener("resize", scheduleUpdate);
    };
  }, [resultRef]);

  if (!placement) return null;

  return createPortal(
    <div
      className="pointer-events-none fixed z-70 -translate-x-1/2"
      style={{ left: placement.left, top: placement.top }}
    >
      <span className="pointer-events-auto inline-flex rounded-2xl border border-border/40 bg-popover/80 backdrop-blur-xl shadow-xl">
        <Button type="button" variant="ghost" size="sm" onClick={onCollapse}>
          <ChevronUp />
          收起结果
        </Button>
      </span>
    </div>,
    document.body,
  );
}

export function ToolResultViewport({
  children,
}: {
  children: (expanded: boolean) => ReactNode;
}) {
  const resultRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  const [canExpand, setCanExpand] = useState(false);

  useEffect(() => {
    const content = contentRef.current;
    if (!content) throw new Error("Tool result viewport has no content");

    const measure = () => {
      const next = content.scrollHeight > COLLAPSED_RESULT_HEIGHT;
      setCanExpand((current) => current === next ? current : next);
    };
    const resizeObserver = new ResizeObserver(measure);
    resizeObserver.observe(content);
    return () => resizeObserver.disconnect();
  }, []);

  const collapse = () => {
    setExpanded(false);
    requestAnimationFrame(() => {
      resultRef.current?.scrollIntoView({ block: "nearest" });
    });
  };

  return (
    <div ref={resultRef} className="relative">
      <div className={cn("relative", !expanded && "max-h-64 overflow-hidden")}>
        <div ref={contentRef}>{children(expanded)}</div>

        {!expanded && canExpand ? (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-20 bg-linear-to-t from-card via-card/90 to-transparent" />
        ) : null}
      </div>

      {!expanded && canExpand ? (
        <div className="pointer-events-none absolute inset-x-0 bottom-2 z-10 flex justify-center">
          <span className="pointer-events-auto inline-flex rounded-xl border border-border/30 bg-popover/70 backdrop-blur-sm shadow-sm">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setExpanded(true)}
              aria-expanded={false}
            >
              <ChevronDown />
              展开工具结果
            </Button>
          </span>
        </div>
      ) : null}

      {expanded && canExpand ? (
        <FloatingCollapseButton resultRef={resultRef} onCollapse={collapse} />
      ) : null}
    </div>
  );
}
