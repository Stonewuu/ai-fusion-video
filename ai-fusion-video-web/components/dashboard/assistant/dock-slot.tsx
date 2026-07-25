"use client";

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, LayoutGroup, motion, useReducedMotion } from "framer-motion";
import { AssistantLauncher } from "./launcher";
import { AssistantWindow } from "./assistant-window";
import { clampDockWidth, getViewportSize } from "./geometry";
import { useAssistantStore, type AssistantMode } from "@/lib/store/assistant-store";
import { useAuthStore } from "@/lib/store/auth-store";

interface AssistantDockSlotProps {
  projectId?: number | null;
}

export function AssistantDockSlot({ projectId }: AssistantDockSlotProps) {
  const reducedMotion = useReducedMotion();
  const userId = useAuthStore((state) => state.user?.id ?? null);
  const mode = useAssistantStore((state) => state.mode);
  const initialized = useAssistantStore((state) => state.initialized);
  const dockWidth = useAssistantStore((state) => state.dockWidth);
  const normalRect = useAssistantStore((state) => state.normalRect);
  const setMode = useAssistantStore((state) => state.setMode);
  const initializeForUser = useAssistantStore((state) => state.initializeForUser);
  const resetForUser = useAssistantStore((state) => state.resetForUser);
  const closeAssistant = useAssistantStore((state) => state.closeAssistant);
  const clampViewportGeometry = useAssistantStore((state) => state.clampViewportGeometry);
  const updateDockWidth = useAssistantStore((state) => state.updateDockWidth);
  const [canDock, setCanDock] = useState(false);
  const [mobileViewport, setMobileViewport] = useState(() =>
    typeof window !== "undefined" && window.innerWidth < 720,
  );
  const [portalHost] = useState<HTMLDivElement | null>(() => {
    if (typeof document === "undefined") return null;
    const host = document.createElement("div");
    host.dataset.slot = "assistant-portal-host";
    return host;
  });
  const dockSlotRef = useRef<HTMLDivElement>(null);
  const dockResizeRef = useRef<{ pointerId: number } | null>(null);
  const previousModeRef = useRef(mode);

  useEffect(() => {
    return () => portalHost?.remove();
  }, [portalHost]);

  useEffect(() => {
    document.body.dataset.assistantLayer = "true";
    return () => {
      delete document.body.dataset.assistantLayer;
    };
  }, []);

  useLayoutEffect(() => {
    if (!portalHost) return;
    const docked = mode === "docked";
    const parent = docked ? dockSlotRef.current : document.body;
    if (!parent) return;
    if (portalHost.parentElement !== parent) parent.appendChild(portalHost);
    Object.assign(portalHost.style, docked
      ? {
          position: "absolute",
          inset: "0",
          width: "100%",
          height: "100%",
          zIndex: "2",
          pointerEvents: "none",
        }
      : {
          position: "fixed",
          inset: "0",
          width: "100%",
          height: "100%",
          zIndex: "65",
          pointerEvents: "none",
        });
  }, [mode, portalHost]);

  useEffect(() => {
    if (userId) initializeForUser(userId);
    else if (initialized) resetForUser();
  }, [initialized, initializeForUser, resetForUser, userId]);

  useEffect(() => () => {
    useAssistantStore.getState().resetForUser();
  }, []);

  useEffect(() => {
    const previousMode = previousModeRef.current;
    previousModeRef.current = mode;
    if (mode === "collapsed" && previousMode !== "collapsed") {
      const frame = requestAnimationFrame(() => {
        document.querySelector<HTMLButtonElement>('[aria-label="打开融光助手"]')?.focus({ preventScroll: true });
      });
      return () => cancelAnimationFrame(frame);
    }
    return undefined;
  }, [mode]);

  useEffect(() => {
    let frame = 0;
    const updateViewport = () => {
      const viewport = getViewportSize();
      const nextMobileViewport = viewport.width < 720;
      const main = dockSlotRef.current?.previousElementSibling as HTMLElement | null;
      const slot = dockSlotRef.current;
      const mainWidth = main?.getBoundingClientRect().width ?? viewport.width;
      const slotWidth = slot?.getBoundingClientRect().width ?? 0;
      // The previous sibling is the real Dashboard main column. Adding the
      // current slot width gives the available row width without inventing a
      // viewport breakpoint or hard-coding the sidebar width.
      const available = Math.max(0, mainWidth + slotWidth);
      const nextCanDock = clampDockWidth(dockWidth, available) > 0;
      setMobileViewport(nextMobileViewport);
      setCanDock(nextCanDock);
      clampViewportGeometry(available);
      const currentMode = useAssistantStore.getState().mode;
      if (nextMobileViewport && currentMode !== "collapsed" && currentMode !== "maximized") {
        setMode("maximized", false);
      } else if (!nextCanDock && currentMode === "docked") {
        setMode("floating", false);
      }
    };
    const scheduleUpdate = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        updateViewport();
      });
    };
    updateViewport();
    const observer = new ResizeObserver(scheduleUpdate);
    const main = dockSlotRef.current?.previousElementSibling;
    if (main) observer.observe(main);
    if (dockSlotRef.current) observer.observe(dockSlotRef.current);
    window.addEventListener("resize", scheduleUpdate);
    return () => {
      if (frame) cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener("resize", scheduleUpdate);
    };
  }, [clampViewportGeometry, dockWidth, initialized, mode, setMode, userId]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      const state = useAssistantStore.getState();
      if (state.mode === "collapsed") return;
      if (state.drawerOpen) {
        state.setDrawerOpen(false);
      } else {
        closeAssistant();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [closeAssistant]);

  useEffect(() => {
    const availableWidth = () => {
      const main = dockSlotRef.current?.previousElementSibling as HTMLElement | null;
      const slot = dockSlotRef.current;
      return Math.max(
        0,
        (main?.getBoundingClientRect().width ?? window.innerWidth)
          + (slot?.getBoundingClientRect().width ?? 0),
      );
    };
    const onMove = (event: PointerEvent) => {
      if (!dockResizeRef.current) return;
      updateDockWidth(window.innerWidth - event.clientX, availableWidth());
    };
    const onUp = () => {
      if (!dockResizeRef.current) return;
      dockResizeRef.current = null;
      const state = useAssistantStore.getState();
      state.updateDockWidth(
        state.dockWidth,
        availableWidth(),
        true,
      );
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
  }, [updateDockWidth]);

  const startDockResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (mode !== "docked" || !canDock) return;
    event.preventDefault();
    event.stopPropagation();
    dockResizeRef.current = { pointerId: event.pointerId };
    document.body.style.userSelect = "none";
    document.body.style.cursor = "ew-resize";
  };

  const toggleDock = () => {
    if (mode === "docked") setMode("floating", canDock);
    else if (canDock) setMode("docked", canDock);
  };

  const toggleMaximize = () => {
    const state = useAssistantStore.getState();
    if (mobileViewport && mode === "maximized") return;
    if (mode === "maximized") state.setMode(state.restoreMode, canDock);
    else state.setMode("maximized", canDock);
  };

  const surfaceClass = mode === "docked"
    ? "relative h-full min-h-0 shrink-0 overflow-visible"
    : "relative h-full w-0 shrink-0 overflow-visible";
  const panelStyle: CSSProperties = mode === "floating"
    ? {
        position: "fixed",
        left: normalRect.x,
        top: normalRect.y,
        width: normalRect.width,
        height: normalRect.height,
        zIndex: 65,
      }
    : mode === "maximized"
      ? {
          position: "fixed",
          inset: "12px",
          width: "auto",
          height: "auto",
          zIndex: 65,
        }
      : {
          position: "relative",
          width: "100%",
          height: "100%",
        };

  if (!initialized || !userId) return null;

  return (
    <LayoutGroup id="fusion-assistant-layout">
      <AnimatePresence initial={false}>
        {mode === "collapsed" ? <AssistantLauncher key="assistant-launcher" canDock={canDock} /> : null}
      </AnimatePresence>
      <motion.div
        ref={dockSlotRef}
        className={surfaceClass}
        animate={{ width: mode === "docked" ? dockWidth : 0 }}
        transition={{ duration: reducedMotion ? 0.01 : mode === "docked" ? 0.42 : 0.32, ease: [0.65, 0, 0.35, 1] }}
      >
        {portalHost && mode !== "collapsed" ? createPortal(
          <section
            className={mode === "docked" ? "h-full w-full overflow-hidden border-l border-border/30 bg-background" : mode === "maximized" ? "overflow-hidden rounded-2xl border border-border/40 bg-popover/80 shadow-xl backdrop-blur-xl" : "h-full w-full overflow-hidden rounded-2xl border border-border/40 bg-popover/80 shadow-xl backdrop-blur-xl"}
            style={{ ...panelStyle, pointerEvents: "auto" }}
          >
            <AssistantWindow
              mode={mode as AssistantMode}
              canDock={canDock}
              mobileViewport={mobileViewport}
              projectId={projectId}
              onToggleDock={toggleDock}
              onToggleMaximize={toggleMaximize}
              onClose={closeAssistant}
              onDockResizeStart={startDockResize}
            />
          </section>,
          portalHost,
        ) : null}
      </motion.div>
    </LayoutGroup>
  );
}
