"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { generationApi } from "@/lib/api/generation";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { useAuthStore } from "@/lib/store/auth-store";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import type { GenerationHistoryEntry, WorkbenchMode } from "./generation-types";
import { isGenerationTaskActive } from "./generation-utils";

interface HistoryState {
  identity: string;
  entries: GenerationHistoryEntry[];
  total: number;
  loading: boolean;
  error: string | null;
}

export function useGenerationHistory(mode: WorkbenchMode) {
  const userId = useAuthStore((state) => state.user?.id);
  const scope = useResourceChangeStore((state) => state.scope);
  const identity = `${userId}:${scope}:${mode}`;
  const generation = useRef(0);
  const lastRequest = useRef(0);
  const polling = useRef(false);
  const [limit, setLimit] = useState(10);
  const [state, setState] = useState<HistoryState | null>(null);
  const [cancelling, setCancelling] = useState<Set<string>>(new Set());
  const current = state?.identity === identity ? state : null;

  const load = useCallback(
    async (size = limit, silent = false) => {
      const version = generation.current;
      const request = ++lastRequest.current;
      if (!silent)
        setState((value) =>
          value?.identity === identity
            ? { ...value, loading: true }
            : {
                identity,
                entries: [],
                total: 0,
                loading: true,
                error: null,
              },
        );
      try {
        const page =
          mode === "image"
            ? await generationApi.pageImageTasks(1, size)
            : await generationApi.pageVideoTasks(1, size);
        const entries = await Promise.all(
          page.list.map(async (task) => ({
            task,
            items:
              mode === "image"
                ? await generationApi.listImageItems(task.id)
                : await generationApi.listVideoItems(task.id),
          })),
        );
        if (version === generation.current && request === lastRequest.current) {
          setState({
            identity,
            entries,
            total: page.total,
            loading: false,
            error: null,
          });
        }
      } catch (error) {
        if (version !== generation.current || request !== lastRequest.current)
          return;
        const denied = [401, 403, 404].includes(getApiErrorStatus(error) ?? 0);
        setState((value) => ({
          identity,
          entries: !denied && value?.identity === identity ? value.entries : [],
          total: !denied && value?.identity === identity ? value.total : 0,
          loading: false,
          error: getApiErrorMessage(error),
        }));
      }
    },
    [identity, limit, mode],
  );

  useEffect(() => {
    const version = ++generation.current;
    void load(limit);
    return () => {
      generation.current = version + 1;
    };
  }, [load, limit]);

  const active = current?.entries.some(({ task }) =>
    isGenerationTaskActive(task),
  );
  useEffect(() => {
    if (!active) return;
    const refresh = () => {
      if (document.visibilityState !== "visible" || polling.current) return;
      polling.current = true;
      void load(limit, true).finally(() => {
        polling.current = false;
      });
    };
    const timer = window.setInterval(refresh, 4000);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refresh);
    };
  }, [active, limit, load]);

  const cancel = async (taskId: string) => {
    const version = generation.current;
    setCancelling((value) => new Set(value).add(taskId));
    try {
      if (mode === "image") await generationApi.cancelImage(taskId);
      else await generationApi.cancelVideo(taskId);
      if (version === generation.current) {
        toast.success("已请求取消，可在执行详情中查看进度");
        await load();
      }
    } catch (error) {
      if (version === generation.current)
        toast.error(getApiErrorMessage(error));
    } finally {
      setCancelling((value) => {
        const next = new Set(value);
        next.delete(taskId);
        return next;
      });
    }
  };

  const remove = async (id: number) => {
    if (mode === "image") await generationApi.deleteImage(id);
    else await generationApi.deleteVideo(id);
    toast.success("创作记录已删除");
    await load();
  };

  return {
    entries: current?.entries ?? [],
    total: current?.total ?? 0,
    loading: current?.loading ?? true,
    error: current?.error ?? null,
    limit,
    setLimit,
    cancelling,
    load,
    cancel,
    remove,
  };
}
