"use client";

import { startTransition, useEffect, useState } from "react";
import type { AiModel } from "@/lib/api/ai-model";
import { useAssistantStore } from "@/lib/store/assistant-store";
import { hasCachedAssistantModels, loadAssistantModels } from "./catalog-cache";

const EMPTY_MODELS: AiModel[] = [];
interface ModelState { userId: number; attempt: number; models: AiModel[]; error: string | null }

export function useAssistantModels(userId: number | null, active: boolean, attempt: number) {
  const [state, setState] = useState<ModelState | null>(null);
  useEffect(() => {
    if (!active || !userId) return;
    let cancelled = false;
    let idleHandle: number | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const beginLoad = () => {
      if (cancelled || useAssistantStore.getState().hydratedUserId !== userId) return;
      void loadAssistantModels(userId).then((models) => {
        const store = useAssistantStore.getState();
        if (cancelled || store.hydratedUserId !== userId) return;
        startTransition(() => {
          setState({ userId, attempt, models, error: null });
          const current = store.selectedModelId;
          const preferred = current && models.some((model) => model.id === current)
            ? current : models.find((model) => model.defaultModel)?.id ?? models[0]?.id ?? null;
          if (preferred !== current) store.setSelectedModelId(preferred);
        });
      }).catch((error: unknown) => {
        if (!cancelled) setState({ userId, attempt, models: EMPTY_MODELS, error: error instanceof Error ? error.message : "模型加载失败" });
      });
    };
    if (attempt > 0 || hasCachedAssistantModels(userId)) beginLoad();
    else if ("requestIdleCallback" in window) idleHandle = window.requestIdleCallback(beginLoad, { timeout: 1200 });
    else timer = setTimeout(beginLoad, 250);
    return () => {
      cancelled = true;
      if (idleHandle !== null) window.cancelIdleCallback(idleHandle);
      if (timer !== null) clearTimeout(timer);
    };
  }, [active, attempt, userId]);
  const current = state?.userId === userId && state.attempt === attempt ? state : null;
  return { models: current?.models ?? EMPTY_MODELS, modelsLoading: !current, modelsError: current?.error ?? null };
}
