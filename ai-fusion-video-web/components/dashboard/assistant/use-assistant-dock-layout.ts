"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { useMotionValue } from "framer-motion";
import { clampDockWidth, getViewportSize } from "./geometry";
import { useAssistantStore, type AssistantMode } from "@/lib/store/assistant-store";

interface UseAssistantDockLayoutOptions {
  mode: AssistantMode;
  dockWidth: number;
  enabled: boolean;
  resizeEnabled: boolean;
}

export function useAssistantDockLayout({
  mode,
  dockWidth,
  enabled,
  resizeEnabled,
}: UseAssistantDockLayoutOptions) {
  const clampViewportGeometry = useAssistantStore((state) => state.clampViewportGeometry);
  const setMode = useAssistantStore((state) => state.setMode);
  const updateDockWidth = useAssistantStore((state) => state.updateDockWidth);
  const [canDock, setCanDock] = useState(false);
  const [mobileViewport, setMobileViewport] = useState(() =>
    typeof window !== "undefined" && window.innerWidth < 650,
  );
  const [viewportSize, setViewportSize] = useState(getViewportSize);
  const dockSlotRef = useRef<HTMLDivElement>(null);
  const resizeRef = useRef<{ liveWidth: number } | null>(null);
  const resizeFrameRef = useRef<number | null>(null);
  const dockSlotWidth = useMotionValue(mode === "docked" ? dockWidth : 0);

  const availableWidth = useCallback(() => {
    const main = dockSlotRef.current?.previousElementSibling as HTMLElement | null;
    const slot = dockSlotRef.current;
    if (!main || !slot) return 0;
    return Math.max(0, main.getBoundingClientRect().width + slot.getBoundingClientRect().width);
  }, []);

  useEffect(() => {
    if (!enabled) return;
    let frame = 0;
    const updateViewport = () => {
      const viewport = getViewportSize();
      const nextMobileViewport = viewport.width < 650;
      const available = availableWidth();
      const nextCanDock = clampDockWidth(dockWidth, available) > 0;
      setMobileViewport(nextMobileViewport);
      setViewportSize(viewport);
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
  }, [availableWidth, clampViewportGeometry, dockWidth, enabled, mode, setMode]);

  useEffect(() => {
    const paintWidth = () => {
      resizeFrameRef.current = null;
      if (resizeRef.current) dockSlotWidth.set(resizeRef.current.liveWidth);
    };
    const onMove = (event: PointerEvent) => {
      const resize = resizeRef.current;
      if (!resize) return;
      const nextWidth = clampDockWidth(window.innerWidth - event.clientX, availableWidth());
      if (!nextWidth) return;
      resize.liveWidth = nextWidth;
      if (resizeFrameRef.current === null) resizeFrameRef.current = requestAnimationFrame(paintWidth);
    };
    const onUp = () => {
      const resize = resizeRef.current;
      if (!resize) return;
      if (resizeFrameRef.current !== null) cancelAnimationFrame(resizeFrameRef.current);
      resizeFrameRef.current = null;
      dockSlotWidth.set(resize.liveWidth);
      resizeRef.current = null;
      updateDockWidth(resize.liveWidth, availableWidth(), true);
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
      if (resizeFrameRef.current !== null) cancelAnimationFrame(resizeFrameRef.current);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
  }, [availableWidth, dockSlotWidth, updateDockWidth]);

  const startResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (mode !== "docked" || !canDock || !resizeEnabled) return;
    event.preventDefault();
    event.stopPropagation();
    resizeRef.current = { liveWidth: dockWidth };
    document.body.style.userSelect = "none";
    document.body.style.cursor = "ew-resize";
  };

  return {
    canDock,
    dockSlotRef,
    dockSlotWidth,
    mobileViewport,
    startResize,
    viewportSize,
  };
}
