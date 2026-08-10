"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type { PipelineTask } from "@/lib/store/pipeline-store";
import { formatElapsed } from "./utils";

export function useElapsed(task: PipelineTask): string {
  const [now, setNow] = useState(() => task.finishedAt ?? Date.now());

  useEffect(() => {
    if (task.status !== "running") return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [task.status]);

  const endTime = task.finishedAt ?? now;
  return formatElapsed(endTime - task.createdAt);
}

export function ElapsedText({ task }: { task: PipelineTask }) {
  const text = useElapsed(task);
  return <>{text}</>;
}

export function useScrollElement() {
  const [element, setElement] = useState<HTMLDivElement | null>(null);
  const ref = useCallback((node: HTMLDivElement | null) => {
    setElement(node);
  }, []);
  return [element, ref] as const;
}

export function useSmartScroll(
  element: HTMLDivElement | null,
  deps: unknown[],
  active: boolean,
) {
  const userScrolledUp = useRef(false);

  const isNearBottom = useCallback(() => {
    if (!element) return true;
    return element.scrollHeight - element.scrollTop - element.clientHeight < 40;
  }, [element]);

  useEffect(() => {
    if (!element) return;
    const onScroll = () => {
      userScrolledUp.current = !isNearBottom();
    };
    element.addEventListener("scroll", onScroll, { passive: true });
    return () => element.removeEventListener("scroll", onScroll);
  }, [element, isNearBottom]);

  useLayoutEffect(() => {
    if (!active || !element) return;
    if (!userScrolledUp.current) {
      element.scrollTop = element.scrollHeight;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, element, ...deps]);
}
