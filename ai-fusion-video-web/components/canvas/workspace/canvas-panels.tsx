"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { CanvasSegmented } from "../ui/canvas-segmented";
import { CanvasPanelPages } from "./panel-pages";
import {
  canvasLeftTabs,
  canvasRightTabs,
  type CanvasLeftPanel,
  type CanvasRightPanel,
  type CanvasMobilePanel,
  type CanvasPanel,
  type CanvasPanelTab,
} from "./panel-navigation";

const EASE = [0.25, 0.46, 0.45, 0.94] as const;

export function CanvasWorkspacePanels({
  left,
  right,
  mobile,
  visited,
  badges = {},
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
  /** 停靠栏角标提示（如运行中、待确认导入或已选连线）。 */
  badges?: Partial<Record<CanvasPanel, "active" | "attention">>;
  onLeft: (value: CanvasLeftPanel) => void;
  onRight: (value: CanvasRightPanel) => void;
  onCloseLeft: () => void;
  onCloseRight: () => void;
  content: Record<CanvasPanel, ReactNode>;
  children: ReactNode;
}) {
  const reduced = useReducedMotion();
  const root = useRef<HTMLDivElement>(null);
  const editor = useRef<HTMLElement>(null);
  const leftAside = useRef<HTMLElement>(null);
  const rightAside = useRef<HTMLElement>(null);
  const mobileReturn = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const narrow = window.matchMedia("(max-width: 767px)");
    const sync = () => {
      const overlayLeft = mobile === "left" && !!left && narrow.matches;
      const overlayRight = mobile === "right" && !!right && narrow.matches;
      if (editor.current) {
        editor.current.inert = overlayLeft || overlayRight;
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
        overlay
          .querySelector<HTMLButtonElement>("[data-panel-close]")
          ?.focus({ preventScroll: true });
      }
    };
    sync();
    narrow.addEventListener("change", sync);
    return () => {
      narrow.removeEventListener("change", sync);
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
      const trigger = root.current?.querySelector<HTMLButtonElement>(
        `button[data-canvas-panel="${panel}"]`,
      );
      if (trigger && trigger.getClientRects().length > 0) {
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

  const openLeft = (value: CanvasLeftPanel) => {
    if (window.innerWidth < 768 && right) {
      onCloseRight();
    }
    onLeft(value);
  };
  const openRight = (value: CanvasRightPanel) => {
    if (window.innerWidth < 768 && left) {
      onCloseLeft();
    }
    onRight(value);
  };

  return (
    <div ref={root} className="relative flex min-h-0 grow basis-0">
      <section
        ref={editor}
        tabIndex={-1}
        className="flex min-h-0 min-w-0 grow basis-0 flex-col outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-label="画布编辑区"
      >
        {children}
      </section>

      {/* 窄屏遮罩：点击空白处收起当前抽屉，桌面端保持画布可同时拖拽交互。 */}
      <AnimatePresence>
        {(left || right) && (
          <motion.div
            key="mobile-backdrop"
            aria-hidden
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: reduced ? 0 : 0.16, ease: EASE }}
            className="absolute inset-0 z-30 bg-background/30 backdrop-blur-[1px] md:hidden"
            onClick={() => {
              if (left) {
                close("left");
              }
              if (right) {
                close("right");
              }
            }}
          />
        )}
      </AnimatePresence>

      <FloatingEdgeDock
        side="left"
        visible={!left}
        tabs={canvasLeftTabs}
        badges={badges}
        reduced={!!reduced}
        label="项目与素材入口"
        onSelect={openLeft}
      />

      <FloatingEdgeDrawer
        side="left"
        active={left}
        asideRef={leftAside}
        tabs={canvasLeftTabs}
        visited={visited}
        content={content}
        reduced={!!reduced}
        label="项目与素材面板"
        navLabel="切换左侧内容"
        showTabIcons
        onSelect={openLeft}
        onClose={() => close("left")}
      />

      <FloatingEdgeDock
        side="right"
        visible={!right}
        tabs={canvasRightTabs}
        badges={badges}
        reduced={!!reduced}
        label="运行、评论与历史入口"
        onSelect={openRight}
      />

      <FloatingEdgeDrawer
        side="right"
        active={right}
        asideRef={rightAside}
        tabs={canvasRightTabs}
        visited={visited}
        content={content}
        reduced={!!reduced}
        label="选区、运行与评审面板"
        navLabel="切换右侧内容"
        showTabIcons={false}
        onSelect={openRight}
        onClose={() => close("right")}
      />
    </div>
  );
}

/** 贴近画布左右边缘居中的轻量悬浮停靠胶囊：收起时不占任何画布宽度，悬停提示各入口。 */
function FloatingEdgeDock<T extends CanvasPanel>({
  side,
  visible,
  tabs,
  badges,
  reduced,
  label,
  onSelect,
}: {
  side: "left" | "right";
  visible: boolean;
  tabs: readonly CanvasPanelTab<T>[];
  badges: Partial<Record<CanvasPanel, "active" | "attention">>;
  reduced: boolean;
  label: string;
  onSelect: (key: T) => void;
}) {
  const offset = side === "left" ? -10 : 10;
  return (
    <AnimatePresence>
      {visible && (
        <motion.nav
          key={`dock-${side}`}
          data-sidebar-navigation={`canvas-${side}-dock`}
          aria-label={label}
          initial={reduced ? false : { opacity: 0, x: offset, scale: 0.94 }}
          animate={{ opacity: 1, x: 0, scale: 1 }}
          exit={reduced ? undefined : { opacity: 0, x: offset, scale: 0.94 }}
          transition={{ duration: reduced ? 0 : 0.18, ease: EASE }}
          className={cn(
            "pointer-events-auto absolute top-1/2 z-30 flex -translate-y-1/2 flex-col items-center gap-1 rounded-[14px] border border-foreground/[0.08] bg-popover/92 p-1 shadow-[0_1px_2px_rgba(0,0,0,0.04),0_8px_24px_-6px_rgba(0,0,0,0.1)] backdrop-blur-xl dark:border-white/[0.1] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_10px_28px_-6px_rgba(0,0,0,0.48)]",
            side === "left" ? "left-3" : "right-3",
          )}
        >
          {tabs.map((tab) => {
            const badge = badges[tab.key];
            return (
              <div key={tab.key} className="relative">
                <CanvasIconButton
                  data-canvas-panel={tab.key}
                  label={tab.title}
                  side={side === "left" ? "right" : "left"}
                  sideOffset={10}
                  onClick={() => onSelect(tab.key)}
                >
                  <tab.icon />
                </CanvasIconButton>
                {badge && (
                  <span
                    aria-hidden
                    className={cn(
                      "pointer-events-none absolute right-1 top-1 size-1.5 rounded-full ring-2 ring-popover",
                      badge === "attention" ? "bg-amber-500" : "bg-primary",
                    )}
                  />
                )}
              </div>
            );
          })}
        </motion.nav>
      )}
    </AnimatePresence>
  );
}

/** 浮于画布之上的毛玻璃抽屉：避开顶部工具栏与底部状态栏，不挤压或重排画布视口。 */
function FloatingEdgeDrawer<T extends CanvasPanel>({
  side,
  active,
  asideRef,
  tabs,
  visited,
  content,
  reduced,
  label,
  navLabel,
  showTabIcons,
  onSelect,
  onClose,
}: {
  side: "left" | "right";
  active: T | null;
  asideRef: React.RefObject<HTMLElement | null>;
  tabs: readonly CanvasPanelTab<T>[];
  visited: CanvasPanel[];
  content: Record<CanvasPanel, ReactNode>;
  reduced: boolean;
  label: string;
  navLabel: string;
  showTabIcons: boolean;
  onSelect: (key: T) => void;
  onClose: () => void;
}) {
  const offset = side === "left" ? -16 : 16;
  const segmentedOptions = tabs.map((tab) => ({
    value: tab.key,
    label: tab.label,
    hint: tab.title,
    icon: showTabIcons ? tab.icon : undefined,
  }));
  return (
    <AnimatePresence>
      {active && (
        <motion.aside
          key={`drawer-${side}`}
          ref={asideRef}
          data-sidebar-navigation={`canvas-${side}`}
          aria-label={label}
          initial={reduced ? false : { opacity: 0, x: offset, scale: 0.985 }}
          animate={{ opacity: 1, x: 0, scale: 1 }}
          exit={reduced ? undefined : { opacity: 0, x: offset, scale: 0.985 }}
          transition={{ duration: reduced ? 0 : 0.22, ease: EASE }}
          onKeyDown={(event) => {
            if (
              event.key === "Escape" &&
              !event.defaultPrevented &&
              !event.nativeEvent.isComposing &&
              event.target instanceof Node &&
              event.currentTarget.contains(event.target)
            ) {
              event.preventDefault();
              onClose();
            }
          }}
          className={cn(
            "canvas-tone-scope pointer-events-auto absolute top-[64px] bottom-[64px] z-40 flex min-h-0 flex-col overflow-hidden rounded-[20px] border border-foreground/[0.09] bg-popover/94 text-popover-foreground shadow-[0_2px_8px_rgba(0,0,0,0.05),0_20px_48px_-12px_rgba(0,0,0,0.16)] backdrop-blur-2xl dark:border-white/[0.11] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.09),0_24px_56px_-12px_rgba(0,0,0,0.62)]",
            side === "left"
              ? "left-3 w-[min(316px,calc(100%-24px))]"
              : "right-3 w-[min(368px,calc(100%-24px))]",
          )}
        >
          <div className="flex shrink-0 items-center gap-2 border-b border-foreground/[0.07] px-3 py-2.5 dark:border-white/[0.08]">
            <CanvasSegmented
              label={navLabel}
              value={active}
              options={segmentedOptions}
              className="flex-1"
              onChange={onSelect}
            />
            <CanvasIconButton
              data-panel-close
              label={`收起${label}`}
              shortcut="Esc"
              side="bottom"
              onClick={onClose}
            >
              <X />
            </CanvasIconButton>
          </div>
          <div className="flex min-h-0 flex-1 flex-col">
            <CanvasPanelPages
              tabs={tabs}
              active={active}
              visited={visited}
              content={content}
            />
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
