"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
  type TouchEvent,
  type UIEvent,
  type WheelEvent,
} from "react";

const AT_BOTTOM_THRESHOLD_PX = 30;

interface UseAssistantMessageScrollOptions {
  contentReady: boolean;
  running: boolean;
  inputHeight: number;
}

export function useAssistantMessageScroll({
  contentReady,
  running,
  inputHeight,
}: UseAssistantMessageScrollOptions) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  const isDetachedRef = useRef(false);
  const isInitializingRef = useRef(true);
  const previousRunningRef = useRef(running);
  const lastScrollTopRef = useRef(0);
  const touchYRef = useRef<number | null>(null);
  const followFrameRef = useRef<number | null>(null);

  const [viewportReady, setViewportReady] = useState(false);
  const [showBackToBottom, setShowBackToBottom] = useState(false);

  const pinToBottom = useCallback(() => {
    const element = viewportRef.current;
    if (!element) return;
    element.scrollTop = element.scrollHeight;
    lastScrollTopRef.current = element.scrollTop;
  }, []);

  const scheduleFollow = useCallback(() => {
    if (followFrameRef.current !== null || isDetachedRef.current) return;
    followFrameRef.current = requestAnimationFrame(() => {
      followFrameRef.current = null;
      if (!isDetachedRef.current) {
        pinToBottom();
      }
    });
  }, [pinToBottom]);

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
      pinToBottom();
    }
  }, [pinToBottom, running]);

  // Re-pin when inputHeight changes if auto-following
  useLayoutEffect(() => {
    if (viewportReady && !isDetachedRef.current) {
      pinToBottom();
    }
  }, [inputHeight, pinToBottom, viewportReady]);

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
  useEffect(() => () => {
    if (followFrameRef.current !== null) {
      cancelAnimationFrame(followFrameRef.current);
    }
  }, []);

  const detachFromBottom = useCallback(() => {
    if (isDetachedRef.current) return;
    isDetachedRef.current = true;
    const element = viewportRef.current;
    if (element) {
      setShowBackToBottom(element.scrollHeight > element.clientHeight + 20);
    }
  }, []);

  const onViewportScroll = useCallback((event: UIEvent<HTMLDivElement>) => {
    if (isInitializingRef.current) return;

    const element = event.currentTarget;
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
    const atBottom = distance <= AT_BOTTOM_THRESHOLD_PX;

    if (atBottom) {
      isDetachedRef.current = false;
      setShowBackToBottom(false);
    } else {
      // Detach only if user manually scrolled upwards (scrollTop decreased)
      if (element.scrollTop < lastScrollTopRef.current - 1) {
        detachFromBottom();
      } else if (isDetachedRef.current) {
        setShowBackToBottom(element.scrollHeight > element.clientHeight + 20);
      }
    }

    lastScrollTopRef.current = element.scrollTop;
  }, [detachFromBottom]);

  const onWheel = useCallback((event: WheelEvent<HTMLDivElement>) => {
    if (event.deltaY < 0) {
      const element = viewportRef.current;
      if (element) {
        const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
        if (distance > 5) {
          detachFromBottom();
        }
      }
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
      if (element) {
        const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
        if (distance > 5) {
          detachFromBottom();
        }
      }
    }
    touchYRef.current = nextY;
  }, [detachFromBottom]);

  const scrollToBottom = useCallback(() => {
    const element = viewportRef.current;
    if (!element) return;
    isDetachedRef.current = false;
    setShowBackToBottom(false);
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    element.scrollTo({
      top: element.scrollHeight,
      behavior: reducedMotion ? "auto" : "smooth",
    });
    lastScrollTopRef.current = element.scrollHeight;
  }, []);

  return {
    viewportRef: viewportRef as RefObject<HTMLDivElement | null>,
    contentRef: contentRef as RefObject<HTMLDivElement | null>,
    viewportReady,
    showBackToBottom,
    onViewportScroll,
    onWheel,
    onTouchStart,
    onTouchMove,
    scrollToBottom,
  };
}

