"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { X, PanelLeftClose, PanelRightClose } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { AnimatedCanvasPanelBody } from "./animated-panel-body";
import {
  CanvasPanelNavigation,
  canvasLeftTabs,
  canvasRightTabs,
  type CanvasLeftPanel,
  type CanvasRightPanel,
  type CanvasMobilePanel,
  type CanvasPanel,
} from "./panel-navigation";

export function CanvasWorkspacePanels({
  left,
  right,
  mobile,
  visited,
  onLeft,
  onRight,
  onCloseLeft,
  onCloseRight,
  content,
  children,
}: {
  left: CanvasLeftPanel | null;
  right: CanvasRightPanel | null;
  mobile: CanvasMobilePanel;
  visited: CanvasPanel[];
  onLeft: (value: CanvasLeftPanel) => void;
  onRight: (value: CanvasRightPanel) => void;
  onCloseLeft: () => void;
  onCloseRight: () => void;
  content: Record<CanvasPanel, ReactNode>;
  children: ReactNode;
}) {
  const editor = useRef<HTMLElement>(null);
  const leftAside = useRef<HTMLElement>(null);
  const rightAside = useRef<HTMLElement>(null);
  const navigation = useRef<HTMLDivElement>(null);
  const mobileReturn = useRef<HTMLElement | null>(null);
  useEffect(() => {
    const narrowLeft = window.matchMedia("(max-width: 767px)");
    const narrowRight = window.matchMedia("(max-width: 1279px)");
    const sync = () => {
      const overlayLeft = mobile === "left" && !!left && narrowLeft.matches;
      const overlayRight = mobile === "right" && !!right && narrowRight.matches;
      if (editor.current) {
        editor.current.inert = overlayLeft || overlayRight;
      }
      if (leftAside.current) {
        leftAside.current.inert = (narrowLeft.matches && !overlayLeft) || overlayRight;
      }
      if (rightAside.current) {
        rightAside.current.inert = (narrowRight.matches && !overlayRight) || overlayLeft;
      }
      const overlay = overlayLeft
        ? leftAside.current
        : overlayRight
          ? rightAside.current
          : null;
      if (overlay && !overlay.contains(document.activeElement)) {
        if (document.activeElement instanceof HTMLElement) {
          mobileReturn.current = document.activeElement;
        }
        overlay.querySelector<HTMLButtonElement>("[data-panel-close]")?.focus({ preventScroll: true });
      }
    };
    sync();
    narrowLeft.addEventListener("change", sync);
    narrowRight.addEventListener("change", sync);
    return () => {
      narrowLeft.removeEventListener("change", sync);
      narrowRight.removeEventListener("change", sync);
    };
  }, [left, right, mobile]);

  const close = (side: "left" | "right") => {
    const panel = side === "left" ? left : right;
    const aside = side === "left" ? leftAside.current : rightAside.current;
    const focusWasInside = aside?.contains(document.activeElement);
    if (side === "left") {
      onCloseLeft();
    } else {
      onCloseRight();
    }
    if (!focusWasInside) {
      return;
    }
    requestAnimationFrame(() => {
      const triggers = navigation.current?.querySelectorAll<HTMLButtonElement>(
        `button[data-canvas-panel="${panel}"]`,
      );
      const trigger = [...(triggers ?? [])].find(
        (button) => button.getClientRects().length > 0,
      );
      if (trigger) {
        trigger.focus({ preventScroll: true });
      } else if (
        mobileReturn.current?.isConnected &&
        mobileReturn.current.getClientRects().length > 0
      ) {
        mobileReturn.current.focus({ preventScroll: true });
      } else {
        editor.current?.focus({ preventScroll: true });
      }
      mobileReturn.current = null;
    });
  };
  return (
    <div ref={navigation} className="flex min-h-0 grow basis-0 flex-col">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-border/20 p-2 xl:hidden">
        <CanvasPanelNavigation
          tabs={canvasLeftTabs}
          active={mobile === "left" ? left : null}
          onSelect={onLeft}
          onClose={() => close("left")}
          className="md:hidden"
          label="项目与素材"
        />
        <CanvasPanelNavigation
          tabs={canvasRightTabs}
          icons={false}
          active={mobile === "right" ? right : null}
          onSelect={onRight}
          onClose={() => close("right")}
          className="ml-auto"
          label="编辑与评审"
        />
      </div>
      <div className="relative flex min-h-0 grow basis-0">
        <aside
          ref={leftAside}
          data-sidebar-navigation="canvas-left"
          aria-label="项目与素材面板"
          onKeyDown={(event) => {
            if (
              event.key === "Escape" &&
              !event.defaultPrevented &&
              event.currentTarget.contains(event.target as Node)
            ) {
              event.preventDefault();
              close("left");
            }
          }}
          className={cn(
            "absolute inset-y-0 left-0 z-40 flex w-full max-w-sm min-h-0 flex-col overflow-hidden border-r border-border/20 bg-card transition-[width,translate,opacity] motion-reduce:transition-none md:static md:z-auto md:max-w-none md:shrink-0",
            left && mobile === "left" ? "translate-x-0 opacity-100" : "pointer-events-none -translate-x-full opacity-0 md:pointer-events-auto md:translate-x-0 md:opacity-100",
            left ? "duration-200 ease-out md:w-64" : "duration-150 ease-in md:w-20",
          )}
        >
          <div
            className={cn(
              "flex shrink-0 items-center gap-2 p-2",
              left && "border-b border-border/20",
            )}
          >
            <span className="text-sm font-medium md:hidden">
              {left === "media" ? "媒体素材" : "项目内容"}
            </span>
            <CanvasPanelNavigation
              tabs={canvasLeftTabs}
              active={left}
              collapsed={!left}
              onSelect={onLeft}
              onClose={() => close("left")}
              className={cn("hidden md:flex", !left && "w-full")}
              label="切换左侧内容"
            />
            {left && (
              <Button
                variant="ghost"
                size="icon"
                className="ml-auto"
                data-panel-close
                aria-label="收起项目与素材面板"
                title="收起左侧面板"
                onClick={() => close("left")}
              >
                <PanelLeftClose className="hidden md:block" />
                <X className="md:hidden" />
              </Button>
            )}
          </div>
          <AnimatedCanvasPanelBody
            side="left"
            tabs={canvasLeftTabs}
            active={left}
            visited={visited}
            content={content}
          />
        </aside>
        <section
          ref={editor}
          tabIndex={-1}
          className="flex min-h-0 min-w-0 grow basis-0 flex-col outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label="画布编辑区"
        >
          {children}
        </section>
        <aside
          ref={rightAside}
          data-sidebar-navigation="canvas-right"
          aria-label="编辑与评审面板"
          onKeyDown={(event) => {
            if (
              event.key === "Escape" &&
              !event.defaultPrevented &&
              event.currentTarget.contains(event.target as Node)
            ) {
              event.preventDefault();
              close("right");
            }
          }}
          className={cn(
            "absolute inset-y-0 right-0 z-50 flex w-full max-w-sm min-h-0 flex-col overflow-hidden border-l border-border/20 bg-card transition-[width,translate,opacity] motion-reduce:transition-none xl:static xl:z-auto xl:max-w-none xl:shrink-0",
            right && mobile === "right" ? "translate-x-0 opacity-100" : "pointer-events-none translate-x-full opacity-0 xl:pointer-events-auto xl:translate-x-0 xl:opacity-100",
            right ? "duration-200 ease-out xl:w-88" : "duration-150 ease-in xl:w-20",
          )}
        >
          <div
            className={cn(
              "flex shrink-0 items-center gap-2 p-2",
              right && "border-b border-border/20",
            )}
          >
            <span className="text-sm font-medium xl:hidden">
              {canvasRightTabs.find((tab) => tab.key === right)?.title}
            </span>
            <CanvasPanelNavigation
              tabs={canvasRightTabs}
              icons={false}
              active={right}
              collapsed={!right}
              onSelect={onRight}
              onClose={() => close("right")}
              className={cn("hidden xl:flex", !right && "w-full")}
              label="切换右侧内容"
            />
            {right && (
              <Button
                variant="ghost"
                size="icon"
                className="ml-auto"
                data-panel-close
                aria-label="收起编辑与评审面板"
                title="收起右侧面板"
                onClick={() => close("right")}
              >
                <PanelRightClose className="hidden xl:block" />
                <X className="xl:hidden" />
              </Button>
            )}
          </div>
          <AnimatedCanvasPanelBody
            side="right"
            tabs={canvasRightTabs}
            active={right}
            visited={visited}
            content={content}
          />
        </aside>
      </div>
    </div>
  );
}
