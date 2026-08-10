"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
  type KeyboardEvent,
  type PointerEvent,
  type TouchEvent,
  type UIEvent,
  type WheelEvent,
} from "react";

const AT_BOTTOM_THRESHOLD_PX = 30;
const FOLLOW_SCROLL_DURATION_MS = 100;

interface UseAssistantMessageScrollOptions {
  contentReady: boolean;
  contentVersion: string;
  running: boolean;
  inputHeight: number;
}

export function useAssistantMessageScroll({
  contentReady,
  contentVersion,
  running,
  inputHeight,
}: UseAssistantMessageScrollOptions) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  const isDetachedRef = useRef(false);
  const isInitializingRef = useRef(true);
  const isScrollingToBottomRef = useRef(false);
  const isScrollbarDraggingRef = useRef(false);
  const previousRunningRef = useRef(running);
  const lastScrollTopRef = useRef(0);
  const touchYRef = useRef<number | null>(null);
  const followFrameRef = useRef<number | null>(null);

  const [viewportReady, setViewportReady] = useState(false);
  const [showBackToBottom, setShowBackToBottom] = useState(false);

  const cancelFollowAnimation = useCallback(() => {
    if (followFrameRef.current !== null) {
      cancelAnimationFrame(followFrameRef.current);
      followFrameRef.current = null;
    }
    isScrollingToBottomRef.current = false;
  }, []);

  const pinToBottom = useCallback(() => {
    const element = viewportRef.current;
    if (!element) return;
    cancelFollowAnimation();
    element.scrollTop = element.scrollHeight;
    lastScrollTopRef.current = element.scrollTop;
  }, [cancelFollowAnimation]);

  const animateToBottom = useCallback(() => {
    const element = viewportRef.current;
    if (!element || isDetachedRef.current) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reducedMotion) {
      pinToBottom();
      return;
    }

    if (followFrameRef.current !== null) return;

    const startTop = element.scrollTop;
    const startedAt = performance.now();
    isScrollingToBottomRef.current = true;

    const step = (now: number) => {
      if (isDetachedRef.current) {
        followFrameRef.current = null;
        isScrollingToBottomRef.current = false;
        return;
      }

      const targetTop = Math.max(0, element.scrollHeight - element.clientHeight);
      const progress = Math.min((now - startedAt) / FOLLOW_SCROLL_DURATION_MS, 1);
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      element.scrollTop = startTop + (targetTop - startTop) * easedProgress;
      lastScrollTopRef.current = element.scrollTop;

      if (progress < 1) {
        followFrameRef.current = requestAnimationFrame(step);
        return;
      }

      element.scrollTop = targetTop;
      lastScrollTopRef.current = element.scrollTop;
      followFrameRef.current = null;
      isScrollingToBottomRef.current = false;
    };

    followFrameRef.current = requestAnimationFrame(step);
  }, [pinToBottom]);

  const scheduleFollow = useCallback(() => {
    if (
      followFrameRef.current !== null
      || isDetachedRef.current
      || (!running && !isInitializingRef.current)
    ) return;
    animateToBottom();
  }, [animateToBottom, running]);

  // Initial render & conversation switch positioning:
  // Pin to bottom while invisible (viewportReady = false), then reveal.
  useLayoutEffect(() => {
    if (!isInitializingRef.current || !contentReady || inputHeight <= 0 || !viewportRef.current) {
      return;
    }

    isDetachedRef.current = false;
    pinToBottom();

    let secondaryFrameId: number | null = null;
    const initialFrameId = requestAnimationFrame(() => {
      if (!isDetachedRef.current) {
        pinToBottom();
      }
      secondaryFrameId = requestAnimationFrame(() => {
        if (!isDetachedRef.current) {
          pinToBottom();
        }
        isInitializingRef.current = false;
        setShowBackToBottom(false);
        setViewportReady(true);
      });
    });

    return () => {
      cancelAnimationFrame(initialFrameId);
      if (secondaryFrameId !== null) {
        cancelAnimationFrame(secondaryFrameId);
      }
    };
  }, [contentReady, inputHeight, pinToBottom]);

  // Handle run state transitions (e.g., user sends a new message or AI starts streaming)
  useLayoutEffect(() => {
    const wasRunning = previousRunningRef.current;
    previousRunningRef.current = running;

    if (running && !wasRunning) {
      isDetachedRef.current = false;
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setShowBackToBottom(false);
      animateToBottom();
    }
  }, [animateToBottom, running]);

  // Re-pin when inputHeight changes if auto-following
  useLayoutEffect(() => {
    if (viewportReady && running && !isDetachedRef.current) {
      animateToBottom();
    }
  }, [animateToBottom, inputHeight, running, viewportReady]);

  // Reconnect replay and history restoration arrive in independent batches.
  // Follow each committed content batch before paint instead of relying only
  // on the initial ready signal or the asynchronous ResizeObserver callback.
  useLayoutEffect(() => {
    if (viewportReady && running && !isDetachedRef.current) {
      animateToBottom();
    }
  }, [animateToBottom, contentVersion, running, viewportReady]);

  // Observe content & viewport resizes for continuous scroll following
  useEffect(() => {
    const content = contentRef.current;
    const viewport = viewportRef.current;
    if (!content || !viewport || typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver(scheduleFollow);
    observer.observe(content);
    observer.observe(viewport);

    return () => observer.disconnect();
  }, [scheduleFollow]);

  // Cleanup pending rAF
  useEffect(() => cancelFollowAnimation, [cancelFollowAnimation]);

  const detachFromBottom = useCallback(() => {
    if (isDetachedRef.current) return;
    isDetachedRef.current = true;
    cancelFollowAnimation();
    const element = viewportRef.current;
    if (element) {
      setShowBackToBottom(element.scrollHeight > element.clientHeight + 20);
    }
  }, [cancelFollowAnimation]);

  const onViewportScroll = useCallback((event: UIEvent<HTMLDivElement>) => {
    if (isInitializingRef.current) return;

    const element = event.currentTarget;
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
    const atBottom = distance <= AT_BOTTOM_THRESHOLD_PX;
    const movedDown = element.scrollTop > lastScrollTopRef.current + 1;

    if (atBottom && (!isDetachedRef.current || movedDown)) {
      isDetachedRef.current = false;
      isScrollingToBottomRef.current = false;
      setShowBackToBottom(false);
    } else if (isDetachedRef.current) {
      setShowBackToBottom(element.scrollHeight > element.clientHeight + 20);
    } else if (isScrollbarDraggingRef.current) {
      detachFromBottom();
    } else if (running && !isScrollingToBottomRef.current) {
      // Browser scroll restoration and reconnect layout changes can move the
      // viewport without user intent. Keep following in that case.
      scheduleFollow();
    }

    lastScrollTopRef.current = element.scrollTop;
  }, [detachFromBottom, running, scheduleFollow]);

  const onWheel = useCallback((event: WheelEvent<HTMLDivElement>) => {
    if (event.deltaY < 0) {
      const element = viewportRef.current;
      if (element && element.scrollHeight > element.clientHeight + 20) {
        detachFromBottom();
      }
    }
  }, [detachFromBottom]);

  const onKeyDown = useCallback((event: KeyboardEvent<HTMLDivElement>) => {
    const scrollsUp = event.key === "ArrowUp"
      || event.key === "PageUp"
      || event.key === "Home"
      || (event.key === " " && event.shiftKey);
    const element = viewportRef.current;
    if (scrollsUp && element && element.scrollHeight > element.clientHeight + 20) {
      detachFromBottom();
    }
  }, [detachFromBottom]);

  const onTouchStart = useCallback((event: TouchEvent<HTMLDivElement>) => {
    touchYRef.current = event.touches[0]?.clientY ?? null;
  }, []);

  const onTouchMove = useCallback((event: TouchEvent<HTMLDivElement>) => {
    const nextY = event.touches[0]?.clientY;
    const previousY = touchYRef.current;
    if (nextY === undefined || previousY === null) return;

    if (nextY > previousY) {
      const element = viewportRef.current;
      if (element && element.scrollHeight > element.clientHeight + 20) {
        detachFromBottom();
      }
    }
    touchYRef.current = nextY;
  }, [detachFromBottom]);

  const onScrollbarPointerDown = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    const element = viewportRef.current;
    if (element && element.scrollHeight > element.clientHeight + 20) {
      isScrollbarDraggingRef.current = true;
    }
  }, []);

  useEffect(() => {
    const stopScrollbarDrag = () => {
      isScrollbarDraggingRef.current = false;
    };
    window.addEventListener("pointerup", stopScrollbarDrag);
    window.addEventListener("pointercancel", stopScrollbarDrag);
    return () => {
      window.removeEventListener("pointerup", stopScrollbarDrag);
      window.removeEventListener("pointercancel", stopScrollbarDrag);
    };
  }, []);

  const scrollToBottom = useCallback(() => {
    const element = viewportRef.current;
    if (!element) return;
    isDetachedRef.current = false;
    setShowBackToBottom(false);
    animateToBottom();
  }, [animateToBottom]);

  return {
    viewportRef: viewportRef as RefObject<HTMLDivElement | null>,
    contentRef: contentRef as RefObject<HTMLDivElement | null>,
    viewportReady,
    showBackToBottom,
    onViewportScroll,
    onWheel,
    onKeyDown,
    onTouchStart,
    onTouchMove,
    onScrollbarPointerDown,
    scrollToBottom,
  };
}
