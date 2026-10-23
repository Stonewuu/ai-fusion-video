"use client";

import { useEffect, useRef, useState } from "react";
import { generationApi } from "@/lib/api/generation";
import {
  generationExecutionApi,
  type GenerationScope,
} from "@/lib/api/generation-execution";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useAuthStore } from "@/lib/store/auth-store";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import {
  clearPendingGeneration,
  generationSubmissionKey,
  readPendingGeneration,
  type PendingGenerationSubmission,
} from "@/lib/generation-submission";
import type { GenerationInput } from "./generation-submit-request";
import type { WorkbenchMode } from "./generation-types";

interface Binding {
  identity: string;
  scope: GenerationScope | null;
  pending: PendingGenerationSubmission | null;
  error: string | null;
  corrupt: boolean;
}

export function useGenerationSubmission(mode: WorkbenchMode) {
  const userId = useAuthStore((state) => state.user?.id);
  const resourceScope = useResourceChangeStore((state) => state.scope);
  const identity = `${userId}:${resourceScope}:${mode}`;
  const epoch = useRef(0);
  const busyRef = useRef(false);
  const [busy, setBusy] = useState(false);
  const [binding, setBinding] = useState<Binding | null>(null);
  const [refresh, setRefresh] = useState(0);
  const current = binding?.identity === identity ? binding : null;

  useEffect(() => {
    const version = ++epoch.current;
    let stopListening: (() => void) | undefined;
    void generationExecutionApi
      .scope()
      .then((scope) => {
        if (version !== epoch.current) return;
        if (scope.userId !== userId)
          throw new Error("账号已变化，请重新读取生成范围。");
        const key = generationSubmissionKey(mode, scope);
        const read = () => {
          if (version !== epoch.current) return;
          try {
            const pending = readPendingGeneration(
              localStorage.getItem(key),
              mode,
              scope,
            );
            setBinding({
              identity,
              scope,
              pending,
              error: null,
              corrupt: false,
            });
          } catch (error) {
            setBinding({
              identity,
              scope,
              pending: null,
              error: getApiErrorMessage(error),
              corrupt: true,
            });
          }
        };
        read();
        const changed = (event: StorageEvent) => {
          if (event.key === key || event.key === null) read();
        };
        window.addEventListener("storage", changed);
        stopListening = () => window.removeEventListener("storage", changed);
      })
      .catch((error) => {
        if (version === epoch.current)
          setBinding({
            identity,
            scope: null,
            pending: null,
            error: getApiErrorMessage(error),
            corrupt: false,
          });
      });
    return () => {
      epoch.current = version + 1;
      stopListening?.();
    };
  }, [identity, userId, mode, refresh]);

  const run = async (
    action: "submit" | "retry" | "withdraw",
    input?: GenerationInput,
    modelName?: string,
  ) => {
    if (busyRef.current || !current?.scope || current.corrupt) return null;
    const scope = current.scope;
    const key = generationSubmissionKey(mode, scope);
    const version = epoch.current;
    busyRef.current = true;
    setBusy(true);
    const publish = (
      pending: PendingGenerationSubmission | null,
      error: string | null = null,
    ) => {
      if (version === epoch.current)
        setBinding({ identity, scope, pending, error, corrupt: false });
    };
    try {
      if (!navigator.locks)
        throw new Error(
          "当前浏览器不支持安全的多窗口提交，请使用新版 Chrome。",
        );
      return await navigator.locks.request(
        key,
        { ifAvailable: true },
        async (lock) => {
          if (!lock)
            throw new Error("另一个窗口正在提交或确认此任务，请等待其完成。");
          let pending = readPendingGeneration(
            localStorage.getItem(key),
            mode,
            scope,
          );
          if (action === "submit") {
            if (pending) {
              publish(pending, "请先确认或撤回上次提交。");
              return null;
            }
            if (!input || !modelName)
              throw new Error("缺少待提交的模型和参数。");
            pending = {
              schemaVersion: 1,
              mode,
              modelName,
              createdAt: new Date().toISOString(),
              request: { ...input, scope, requestId: crypto.randomUUID() },
            };
            // 必须先持久化原始请求；失败时不发送，也不换用临时请求 ID。
            localStorage.setItem(key, JSON.stringify(pending));
          }
          if (!pending) {
            publish(null);
            return null;
          }
          publish(pending);
          try {
            const receipt =
              action === "withdraw"
                ? await generationExecutionApi.withdraw(
                    pending.request.requestId,
                    mode === "image" ? "IMAGE" : "VIDEO",
                    scope,
                  )
                : {
                    taskId:
                      mode === "image"
                        ? await generationApi.submitImage(pending.request)
                        : await generationApi.submitVideo(pending.request),
                    state: "ACCEPTED",
                  };
            try {
              clearPendingGeneration(
                localStorage,
                key,
                pending.request.requestId,
              );
              publish(null);
            } catch (error) {
              publish(
                pending,
                `服务端已确认，但本地记录未能清理：${getApiErrorMessage(error)}。再次确认会读取同一任务。`,
              );
            }
            return version === epoch.current ? receipt : null;
          } catch (error) {
            publish(pending, getApiErrorMessage(error));
            return null;
          }
        },
      );
    } catch (error) {
      if (version === epoch.current)
        setBinding((value) =>
          value?.identity === identity
            ? { ...value, error: getApiErrorMessage(error) }
            : value,
        );
      return null;
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };

  const clearCorruptRecord = async () => {
    if (!current?.scope || !current.corrupt || busyRef.current) return;
    const key = generationSubmissionKey(mode, current.scope);
    const version = epoch.current;
    try {
      if (!navigator.locks)
        throw new Error(
          "当前浏览器不支持安全的多窗口操作，请使用新版 Chrome。",
        );
      await navigator.locks.request(
        key,
        { ifAvailable: true },
        async (lock) => {
          if (!lock) throw new Error("另一个窗口仍在处理此提交，请稍后重试。");
          localStorage.removeItem(key);
        },
      );
      if (version === epoch.current) setRefresh((value) => value + 1);
    } catch (error) {
      if (version === epoch.current)
        setBinding((value) =>
          value ? { ...value, error: getApiErrorMessage(error) } : value,
        );
    }
  };

  return {
    busy,
    pending: current?.pending ?? null,
    error: current?.error ?? null,
    corrupt: current?.corrupt ?? false,
    ready: !!current?.scope && !current.corrupt,
    canSubmit:
      !!current?.scope && !current.pending && !current.corrupt && !busy,
    submit: (input: GenerationInput, modelName: string) =>
      run("submit", input, modelName),
    retry: () => run("retry"),
    withdraw: () => run("withdraw"),
    reload: () => setRefresh((value) => value + 1),
    clearCorruptRecord,
  };
}
