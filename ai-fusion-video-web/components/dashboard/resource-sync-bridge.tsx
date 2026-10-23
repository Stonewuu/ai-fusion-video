"use client";

import { useEffect, useState } from "react";
import { readResourceEvents } from "@/lib/api/resource-events";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import { useAuthStore } from "@/lib/store/auth-store";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { Button } from "@/components/ui/button";

export function ResourceSyncBridge({ enabled }: { enabled: boolean }) {
  const userId = useAuthStore((state) => state.user?.id);
  const [failure, setFailure] = useState<string | null>(null);
  const [attempt, retry] = useState(0);

  useEffect(() => {
    if (!enabled || !userId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let cursor: number | undefined;
    let scope: string | undefined;
    let running = false;
    let failures = 0;
    const poll = async () => {
      if (controller.signal.aborted || running) return;
      running = true;
      let delay = document.hidden ? 15000 : 1000;
      try {
        const page = await readResourceEvents(cursor, scope, controller.signal);
        if (controller.signal.aborted) return;
        useResourceChangeStore.getState().apply(page);
        cursor = page.cursor;
        scope = page.scope;
        failures = 0;
        setFailure(null);
        if (page.hasMore) delay = 0;
      } catch (error) {
        if (controller.signal.aborted) return;
        setFailure(getApiErrorMessage(error));
        failures += 1;
        delay = Math.min(15000, 1000 * 2 ** Math.min(failures, 4));
      } finally {
        running = false;
        if (!controller.signal.aborted) timer = setTimeout(poll, delay);
      }
    };
    const wake = () => {
      if (document.hidden || running) return;
      clearTimeout(timer);
      void poll();
    };
    timer = setTimeout(poll, 0);
    document.addEventListener("visibilitychange", wake);
    window.addEventListener("online", wake);
    return () => {
      controller.abort();
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", wake);
      window.removeEventListener("online", wake);
    };
  }, [enabled, userId, attempt]);

  if (!enabled || !failure) return null;
  return (
    <div role="status" className="fixed bottom-3 left-1/2 z-50 flex max-w-full -translate-x-1/2 items-center gap-3 rounded-2xl border border-border/40 bg-popover/80 p-3 shadow-xl backdrop-blur-xl">
      <span className="text-sm text-muted-foreground" title={failure}>资源同步中断，正在重连</span>
      <Button variant="outline" size="sm" onClick={() => retry((value) => value + 1)}>重试</Button>
    </div>
  );
}
