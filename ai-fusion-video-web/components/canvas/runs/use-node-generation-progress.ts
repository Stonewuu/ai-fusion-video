"use client";

import { useEffect, useState } from "react";
import { generationExecutionApi } from "@/lib/api/generation-execution";
import type { CanvasNodeActivity } from "@/lib/api/canvas-runs";
import { generationNodeProgress } from "@/lib/canvas/node-execution";
import { runFinished } from "@/lib/canvas/run-state";

export function useNodeGenerationProgress(activity?: CanvasNodeActivity) {
  const node = activity?.node;
  const id = node?.externalKind === "GENERATION" && !node.contentUnavailable && activity?.canControl ? node.externalId : null;
  const active = !!node && !runFinished(node.status);
  const [observed, setObserved] = useState<{ id: string; value: ReturnType<typeof generationNodeProgress> | null; unavailable: boolean } | null>(null);
  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const read = async () => {
      try {
        const result = await generationExecutionApi.get(id, controller.signal);
        if (!controller.signal.aborted) setObserved({ id, value: generationNodeProgress(result), unavailable: false });
      } catch {
        if (!controller.signal.aborted) setObserved((current) => ({ id, value: current?.id === id ? current.value : null, unavailable: true }));
      } finally {
        if (!controller.signal.aborted && active) timer = setTimeout(() => void read(), 2500);
      }
    };
    void read();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [id, active]);
  return id && observed?.id === id ? observed : null;
}
