"use client";

import { useEffect, useRef } from "react";
import { useStore } from "zustand";
import type { CanvasStore } from "@/lib/store/canvas-store";

export function useCanvasSaveRecovery(store: CanvasStore) {
  const pending = useStore(store, (state) => state.pending);
  const saving = useStore(store, (state) => state.saving);
  const error = useStore(store, (state) => state.saveError);
  const attempts = useRef(0);
  useEffect(() => {
    if (!pending) { attempts.current = 0; return; }
    if (saving || !error) return;
    const recover = () => {
      const state = store.getState();
      if (navigator.onLine && state.pending === pending && !state.saving && state.sync !== "denied")
        void state.retry().catch(() => undefined);
    };
    const timer = window.setTimeout(recover, Math.min(15000, 1000 * 2 ** Math.min(attempts.current++, 4)));
    window.addEventListener("online", recover);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("online", recover);
    };
  }, [error, pending, saving, store]);
}
