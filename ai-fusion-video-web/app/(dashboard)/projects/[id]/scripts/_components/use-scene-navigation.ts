"use client";

import { useCallback, useEffect, useLayoutEffect, useReducer, useRef } from "react";
import { useReducedMotion } from "framer-motion";
import type { SceneItem } from "@/lib/api/script";
import { animateSceneScroll, SCENE_SCROLL_DURATION_MS } from "./scene-scroll";

export function useSceneNavigation(scenes: SceneItem[] | undefined, onSelect: (id: number) => void, ready: boolean,
  loading: boolean) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const isAutoScrollingRef = useRef(false);
  const pendingScene = useRef<number | null>(null);
  const stopAnimation = useRef<(() => void) | null>(null);
  const [requestVersion, requestRender] = useReducer((version: number) => version + 1, 0);
  const reducedMotion = useReducedMotion();

  const cancelSceneScroll = useCallback(() => {
    pendingScene.current = null;
    stopAnimation.current?.();
    stopAnimation.current = null;
    isAutoScrollingRef.current = false;
  }, []);

  const scrollToScene = useCallback((id: number) => {
    cancelSceneScroll();
    pendingScene.current = id;
    isAutoScrollingRef.current = true;
    onSelect(id);
    requestRender();
  }, [cancelSceneScroll, onSelect]);

  useLayoutEffect(() => {
    const container = scrollContainerRef.current;
    if (!ready || loading || !container || pendingScene.current === null) {
      return;
    }
    const scene = container.querySelector<HTMLElement>(`[data-scene-id="${pendingScene.current}"]`);
    if (!scene) {
      return;
    }
    pendingScene.current = null;
    stopAnimation.current = animateSceneScroll(container, scene, reducedMotion ? 0 : SCENE_SCROLL_DURATION_MS, () => {
      isAutoScrollingRef.current = false;
      stopAnimation.current = null;
    });
  }, [requestVersion, scenes, ready, loading, reducedMotion]);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!ready || !container) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) {
        cancelSceneScroll();
      }
    };
    container.addEventListener("wheel", cancelSceneScroll, { passive: true });
    container.addEventListener("touchstart", cancelSceneScroll, { passive: true });
    container.addEventListener("pointerdown", cancelSceneScroll);
    container.addEventListener("keydown", onKeyDown);
    return () => {
      container.removeEventListener("wheel", cancelSceneScroll);
      container.removeEventListener("touchstart", cancelSceneScroll);
      container.removeEventListener("pointerdown", cancelSceneScroll);
      container.removeEventListener("keydown", onKeyDown);
      cancelSceneScroll();
    };
  }, [ready, cancelSceneScroll]);

  return { scrollContainerRef, isAutoScrollingRef, scrollToScene, cancelSceneScroll };
}
