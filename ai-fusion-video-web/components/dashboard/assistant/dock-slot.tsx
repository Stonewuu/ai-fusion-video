"use client";

import {
  startTransition,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion, useMotionValue, useReducedMotion } from "framer-motion";
import { AssistantLauncher, type AssistantLauncherStatus } from "./launcher";
import { subscribeToAssistantOpenRequest } from "./open-assistant";
import { AssistantWindow } from "./assistant-window";
import { useAssistantDockLayout } from "./use-assistant-dock-layout";
import {
  ASSISTANT_LAUNCHER_SIZE,
  type AssistantPoint,
  type AssistantRect,
} from "./geometry";
import { useAssistantStore, type AssistantMode } from "@/lib/store/assistant-store";
import { useAuthStore } from "@/lib/store/auth-store";
import { cn } from "@/lib/utils";

interface AssistantDockSlotProps {
  projectId?: number | null;
}

type OverlayMode = Exclude<AssistantMode, "docked">;
type DockTransition =
  | { phase: "idle" }
  | { phase: "overlay-morph-in" }
  | { phase: "overlay-out" }
  | { phase: "dock-in" }
  | { phase: "dock-out"; target: OverlayMode }
  | { phase: "overlay-in" }
  | { phase: "overlay-resize" };

const PAGE_EASE = [0.65, 0, 0.35, 1] as const;

function isRunningStatus(status: string | undefined) {
  return status === "running"
    || status === "pending"
    || status === "RUNNING"
    || status === "WAITING_CONFIRMATION"
    || status === "WAITING_EXTERNAL"
    || status === "CANCEL_REQUESTED";
}

function overlayGeometry(
  mode: OverlayMode,
  launcher: AssistantPoint,
  normal: AssistantRect,
  viewport: { width: number; height: number },
) {
  if (mode === "collapsed") {
    return {
      left: launcher.x,
      top: launcher.y,
      width: ASSISTANT_LAUNCHER_SIZE,
      height: ASSISTANT_LAUNCHER_SIZE,
      borderRadius: ASSISTANT_LAUNCHER_SIZE / 2,
    };
  }
  if (mode === "floating") {
    return {
      left: normal.x,
      top: normal.y,
      width: normal.width,
      height: normal.height,
      borderRadius: 18,
    };
  }
  return {
    left: 12,
    top: 12,
    width: viewport.width - 24,
    height: viewport.height - 24,
    borderRadius: 18,
  };
}

export function AssistantDockSlot({ projectId }: AssistantDockSlotProps) {
  const reducedMotion = useReducedMotion();
  const userId = useAuthStore((state) => state.user?.id ?? null);
  const mode = useAssistantStore((state) => state.mode);
  const initialized = useAssistantStore((state) => state.initialized);
  const dockWidth = useAssistantStore((state) => state.dockWidth);
  const normalRect = useAssistantStore((state) => state.normalRect);
  const launcherPosition = useAssistantStore((state) => state.launcherPosition);
  const launcherStatus = useAssistantStore((state): AssistantLauncherStatus => {
    const runtimes = Object.values(state.conversationStates);
    if (runtimes.some((runtime) => isRunningStatus(runtime.status))) return "running";
    if (runtimes.some((runtime) => runtime.unread)) return "completed";
    return "idle";
  });
  const initializeForUser = useAssistantStore((state) => state.initializeForUser);
  const resetForUser = useAssistantStore((state) => state.resetForUser);
  const closeAssistant = useAssistantStore((state) => state.closeAssistant);
  const [dockTransition, setDockTransition] = useState<DockTransition>({ phase: "idle" });
  const [contentReadyUserId, setContentReadyUserId] = useState<number | null>(null);
  const [overlayContentReady, setOverlayContentReady] = useState(mode !== "collapsed");
  const [heavyContentReady, setHeavyContentReady] = useState(false);
  const [launcherWavesVisible, setLauncherWavesVisible] = useState(mode === "collapsed");
  const [pointerReady, setPointerReady] = useState(true);
  const overlayContentRevealTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const previousModeRef = useRef(mode);
  const {
    canDock,
    dockSlotRef,
    dockSlotWidth,
    mobileViewport,
    startResize: startDockResize,
    viewportSize,
  } = useAssistantDockLayout({
    mode,
    dockWidth,
    enabled: initialized && !!userId,
    resizeEnabled: dockTransition.phase === "idle",
  });
  const surfaceLeft = useMotionValue(launcherPosition.x);
  const surfaceTop = useMotionValue(launcherPosition.y);
  const surfaceWidth = useMotionValue<number | string>(ASSISTANT_LAUNCHER_SIZE);
  const surfaceHeight = useMotionValue<number | string>(ASSISTANT_LAUNCHER_SIZE);
  const surfaceRadius = useMotionValue(ASSISTANT_LAUNCHER_SIZE / 2);

  const paintLauncherPosition = useCallback((position: AssistantPoint) => {
    surfaceLeft.set(position.x);
    surfaceTop.set(position.y);
  }, [surfaceLeft, surfaceTop]);

  const paintWindowRect = useCallback((rect: AssistantRect) => {
    surfaceLeft.set(rect.x);
    surfaceTop.set(rect.y);
    surfaceWidth.set(rect.width);
    surfaceHeight.set(rect.height);
  }, [surfaceHeight, surfaceLeft, surfaceTop, surfaceWidth]);

  const beginDockExit = useCallback((target: OverlayMode) => {
    if (dockTransition.phase !== "idle") return;
    setHeavyContentReady(false);
    setDockTransition({ phase: "dock-out", target });
  }, [dockTransition.phase]);

  const openAssistantWindow = useCallback((openedByPointer: boolean) => {
    if (dockTransition.phase !== "idle") return;
    if (overlayContentRevealTimerRef.current !== null) {
      clearTimeout(overlayContentRevealTimerRef.current);
      overlayContentRevealTimerRef.current = null;
    }
    setOverlayContentReady(false);
    setHeavyContentReady(false);
    setLauncherWavesVisible(false);
    setPointerReady(!openedByPointer);
    const state = useAssistantStore.getState();
    const desired = mobileViewport
      ? "maximized"
      : state.lastOpenMode === "maximized"
        ? state.restoreMode
        : state.lastOpenMode;
    const target = desired === "docked" && !canDock ? "floating" : desired;
    if (target === "docked") setDockTransition({ phase: "overlay-out" });
    else {
      setDockTransition({ phase: "overlay-morph-in" });
      state.openAssistant(canDock);
      if (reducedMotion) {
        startTransition(() => setOverlayContentReady(true));
      } else {
        overlayContentRevealTimerRef.current = setTimeout(() => {
          overlayContentRevealTimerRef.current = null;
          startTransition(() => setOverlayContentReady(true));
        }, 32);
      }
    }
  }, [canDock, dockTransition.phase, mobileViewport, reducedMotion]);

  useEffect(() => subscribeToAssistantOpenRequest(() => {
    if (useAssistantStore.getState().mode === "collapsed") {
      openAssistantWindow(false);
    }
  }), [openAssistantWindow]);

  const closeAssistantWindow = useCallback(() => {
    if (overlayContentRevealTimerRef.current !== null) {
      clearTimeout(overlayContentRevealTimerRef.current);
      overlayContentRevealTimerRef.current = null;
    }
    setOverlayContentReady(false);
    setHeavyContentReady(false);
    setLauncherWavesVisible(false);
    setPointerReady(true);
    if (mode === "docked") beginDockExit("collapsed");
    else closeAssistant();
  }, [beginDockExit, closeAssistant, mode]);

  const releaseInteractionGuard = useCallback(() => {
    if (dockTransition.phase !== "idle") return;
    setPointerReady(true);
  }, [dockTransition.phase]);

  const toggleDock = useCallback(() => {
    if (dockTransition.phase !== "idle") return;
    if (mode === "docked") beginDockExit("floating");
    else if (canDock) {
      setHeavyContentReady(false);
      setDockTransition({ phase: "overlay-out" });
    }
  }, [beginDockExit, canDock, dockTransition.phase, mode]);

  const toggleMaximize = useCallback(() => {
    if (dockTransition.phase !== "idle") return;
    const state = useAssistantStore.getState();
    if (mobileViewport && mode === "maximized") return;
    if (mode === "docked") {
      beginDockExit("maximized");
    } else if (mode === "maximized" && state.restoreMode === "docked" && canDock) {
      setHeavyContentReady(false);
      setDockTransition({ phase: "overlay-out" });
    } else if (mode === "maximized") {
      setHeavyContentReady(false);
      setDockTransition({ phase: "overlay-resize" });
      state.setMode(state.restoreMode, canDock);
    } else {
      setHeavyContentReady(false);
      setDockTransition({ phase: "overlay-resize" });
      state.setMode("maximized", canDock);
    }
  }, [beginDockExit, canDock, dockTransition.phase, mobileViewport, mode]);

  const finishOverlayAnimation = useCallback(() => {
    if (mode === "collapsed") setLauncherWavesVisible(true);
    if (dockTransition.phase === "overlay-out") {
      setDockTransition({ phase: "dock-in" });
      useAssistantStore.getState().setMode("docked", canDock);
    } else if (
      dockTransition.phase === "overlay-in"
      || dockTransition.phase === "overlay-morph-in"
      || dockTransition.phase === "overlay-resize"
    ) {
      if (overlayContentRevealTimerRef.current !== null) {
        clearTimeout(overlayContentRevealTimerRef.current);
        overlayContentRevealTimerRef.current = null;
      }
      startTransition(() => {
        setOverlayContentReady(true);
        setDockTransition({ phase: "idle" });
      });
    }
  }, [canDock, dockTransition.phase, mode]);

  const finishDockAnimation = useCallback(() => {
    if (dockTransition.phase === "dock-in") {
      setDockTransition({ phase: "idle" });
      return;
    }
    if (dockTransition.phase !== "dock-out") return;
    const target = dockTransition.target;
    setDockTransition({ phase: "overlay-in" });
    useAssistantStore.getState().setMode(target, canDock);
  }, [canDock, dockTransition]);

  useEffect(() => {
    document.body.dataset.assistantLayer = "true";
    return () => {
      delete document.body.dataset.assistantLayer;
    };
  }, []);

  useEffect(() => {
    if (!userId) {
      if (initialized) resetForUser();
      return;
    }

    initializeForUser(userId);
    let idleHandle: number | null = null;
    const frame = requestAnimationFrame(() => {
      const restoredMode = useAssistantStore.getState().mode;
      const markReady = () => {
        startTransition(() => {
          setContentReadyUserId(userId);
          if (restoredMode !== "collapsed" && restoredMode !== "docked") {
            setOverlayContentReady(true);
          }
        });
      };
      if (restoredMode !== "collapsed" && "requestIdleCallback" in window) {
        idleHandle = window.requestIdleCallback(markReady, { timeout: 300 });
      } else {
        markReady();
      }
    });

    return () => {
      cancelAnimationFrame(frame);
      if (idleHandle !== null) window.cancelIdleCallback(idleHandle);
    };
  }, [initialized, initializeForUser, resetForUser, userId]);

  useEffect(() => () => {
    if (overlayContentRevealTimerRef.current !== null) {
      clearTimeout(overlayContentRevealTimerRef.current);
    }
    useAssistantStore.getState().resetForUser();
  }, []);

  useEffect(() => {
    if (mode !== "maximized") return;
    const previousOverflow = document.body.style.overflow;
    const previousOverscrollBehavior = document.body.style.overscrollBehavior;
    document.body.style.overflow = "hidden";
    document.body.style.overscrollBehavior = "none";
    return () => {
      document.body.style.overflow = previousOverflow;
      document.body.style.overscrollBehavior = previousOverscrollBehavior;
    };
  }, [mode]);

  useEffect(() => {
    if (
      heavyContentReady
      || contentReadyUserId !== userId
      || mode === "collapsed"
      || dockTransition.phase !== "idle"
    ) return;

    let secondFrame: number | null = null;
    const firstFrame = requestAnimationFrame(() => {
      secondFrame = requestAnimationFrame(() => {
        startTransition(() => setHeavyContentReady(true));
      });
    });
    return () => {
      cancelAnimationFrame(firstFrame);
      if (secondFrame !== null) cancelAnimationFrame(secondFrame);
    };
  }, [contentReadyUserId, dockTransition.phase, heavyContentReady, mode, userId]);

  useEffect(() => {
    const previousMode = previousModeRef.current;
    previousModeRef.current = mode;
    if (mode !== "collapsed" || previousMode === "collapsed") return;
    const frame = requestAnimationFrame(() => {
      document.querySelector<HTMLButtonElement>('[aria-label="打开融光助手"]')?.focus({ preventScroll: true });
    });
    return () => cancelAnimationFrame(frame);
  }, [mode]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      const state = useAssistantStore.getState();
      if (state.mode === "collapsed") return;
      if (state.drawerOpen) state.setDrawerOpen(false);
      else closeAssistantWindow();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [closeAssistantWindow]);

  const overlayMode = mode === "docked" ? null : mode;
  const overlayTarget = overlayMode
    ? overlayGeometry(overlayMode, launcherPosition, normalRect, viewportSize)
    : null;
  const overlayTransitioning = dockTransition.phase === "overlay-out" || dockTransition.phase === "overlay-in";
  const dockVisible = mode === "docked" && dockTransition.phase !== "dock-out";
  const contentHydrated = contentReadyUserId === userId;
  const dockContentVisible = contentHydrated;
  const overlayContentVisible = contentHydrated && overlayContentReady;
  const conversationContentActive = heavyContentReady && dockTransition.phase === "idle";
  const interactionGuardActive = mode !== "collapsed"
    && (dockTransition.phase !== "idle" || !pointerReady);
  if (!initialized || !userId) return null;

  return (
    <motion.div
      ref={dockSlotRef}
      className={cn(
        "relative h-full min-h-0 shrink-0",
        mode === "docked" && dockTransition.phase === "idle" ? "overflow-visible" : "overflow-hidden",
      )}
      style={{ width: dockSlotWidth }}
      animate={{ width: dockVisible ? dockWidth : 0 }}
      transition={{
        duration: reducedMotion ? 0.01 : dockTransition.phase === "dock-out" ? 0.3 : 0.42,
        ease: PAGE_EASE,
      }}
      onAnimationComplete={finishDockAnimation}
    >
      {mode === "docked" ? (
        <motion.section
          data-assistant-dock-surface
          initial={dockTransition.phase === "dock-in" ? { opacity: 0 } : false}
          animate={{ opacity: dockTransition.phase === "dock-out" ? 0 : 1 }}
          transition={{
            duration: reducedMotion ? 0.01 : dockTransition.phase === "dock-out" ? 0.1 : 0.14,
            delay: reducedMotion || dockTransition.phase === "dock-out" ? 0 : 0.24,
          }}
          className="h-full w-full overflow-hidden border-l border-border/30 bg-background"
        >
          {dockContentVisible ? (
            <AssistantWindow
              mode="docked"
              canDock={canDock}
              mobileViewport={mobileViewport}
              showLeadingIcon
              projectId={projectId}
              onToggleDock={toggleDock}
              onToggleMaximize={toggleMaximize}
              onClose={closeAssistantWindow}
              onRectPaint={paintWindowRect}
              interactionGuardActive={interactionGuardActive}
              onInteractionGuardRelease={releaseInteractionGuard}
              contentActive={conversationContentActive}
              onDockResizeStart={startDockResize}
            />
          ) : null}
        </motion.section>
      ) : null}

      {overlayMode && overlayTarget && typeof document !== "undefined" ? createPortal(
        <>
          <AnimatePresence initial={false}>
            {overlayMode === "maximized" ? (
              <motion.div
                key="assistant-maximized-backdrop"
                data-assistant-backdrop
                aria-hidden="true"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: reducedMotion ? 0.01 : 0.2, ease: "easeOut" }}
                className="modal-overlay fixed inset-0 z-60 touch-none overscroll-none"
              />
            ) : null}
          </AnimatePresence>
          <motion.section
          data-assistant-surface
          data-assistant-launcher-status={overlayMode === "collapsed" ? launcherStatus : undefined}
          data-assistant-transition={dockTransition.phase}
          data-assistant-content-active={conversationContentActive ? "true" : "false"}
          initial={dockTransition.phase === "overlay-in" ? { opacity: 0 } : false}
          animate={{
            ...overlayTarget,
            opacity: dockTransition.phase === "overlay-out" ? 0 : 1,
          }}
          style={{
            left: surfaceLeft,
            top: surfaceTop,
            width: surfaceWidth,
            height: surfaceHeight,
            borderRadius: surfaceRadius,
            pointerEvents: "auto",
          }}
          transition={{
            duration: overlayTransitioning
              ? 0
              : reducedMotion
                ? 0.01
                : overlayMode === "collapsed"
                  ? 0.3
                  : 0.38,
            delay: reducedMotion || overlayTransitioning ? 0 : overlayMode === "collapsed" ? 0.1 : 0,
            ease: PAGE_EASE,
            opacity: {
              duration: reducedMotion ? 0.01 : dockTransition.phase === "overlay-in" ? 0.18 : 0.12,
              delay: 0,
              ease: "easeOut",
            },
          }}
          onAnimationComplete={finishOverlayAnimation}
          className={cn(
            "fixed z-65 border border-border/40 backdrop-blur-xl backdrop-saturate-150",
            overlayMode === "collapsed"
              ? "overflow-visible bg-background/75 shadow-lg"
              : "overflow-hidden bg-popover/80 shadow-xl",
          )}
        >
          <AssistantLauncher
            collapsed={overlayMode === "collapsed"}
            wavesVisible={launcherWavesVisible}
            status={launcherStatus}
            onOpen={openAssistantWindow}
            onPositionPaint={paintLauncherPosition}
          />
          <AnimatePresence initial={false}>
            {overlayMode !== "collapsed" && overlayContentVisible ? (
              <motion.div
                key="assistant-window-content"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{
                  opacity: 0,
                  transition: { duration: reducedMotion ? 0.01 : 0.1, delay: 0 },
                }}
                transition={{
                  duration: reducedMotion ? 0.01 : 0.14,
                  delay: 0,
                }}
                style={dockTransition.phase === "overlay-morph-in" ? {
                  width: overlayTarget.width,
                  height: overlayTarget.height,
                } : undefined}
                className={cn(
                  "absolute overflow-hidden",
                  dockTransition.phase === "overlay-morph-in" ? "left-0 top-0" : "inset-0",
                  dockTransition.phase !== "idle" && "pointer-events-none",
                )}
              >
                <AssistantWindow
                  mode={overlayMode}
                  canDock={canDock}
                  mobileViewport={mobileViewport}
                  showLeadingIcon={false}
                  projectId={projectId}
                  onToggleDock={toggleDock}
                  onToggleMaximize={toggleMaximize}
                  onClose={closeAssistantWindow}
                  onRectPaint={paintWindowRect}
                  interactionGuardActive={interactionGuardActive}
                  onInteractionGuardRelease={releaseInteractionGuard}
                  contentActive={conversationContentActive}
                />
              </motion.div>
            ) : null}
          </AnimatePresence>
          </motion.section>
        </>,
        document.body,
      ) : null}
    </motion.div>
  );
}
