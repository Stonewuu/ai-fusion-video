"use client";

import { useEffect, useRef, useState } from "react";
import {
  comfyUiWorkflowApi,
  type ComfyUiTestAction,
  type ComfyUiTestRequest,
  type ComfyUiTestRun,
} from "@/lib/api/comfyui-workflow";
import {
  generationExecutionApi,
  type GenerationScope,
} from "@/lib/api/generation-execution";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { useAuthStore } from "@/lib/store/auth-store";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import {
  comfyUiTestActive,
  comfyUiTestKey,
  readPendingComfyUiTest,
  type PendingComfyUiTest,
} from "@/lib/comfyui-test-request";

interface Binding {
  identity: string;
  scope: GenerationScope;
  runs: ComfyUiTestRun[];
  pending: PendingComfyUiTest | null;
  corrupt: boolean;
}

export function useComfyUiTests(
  versionId: number,
  onUpdated: () => void | Promise<void>,
) {
  const userId = useAuthStore((state) => state.user?.id);
  const resourceScope = useResourceChangeStore((state) => state.scope);
  const identity = `${userId}:${resourceScope}:${versionId}`;
  const [binding, setBinding] = useState<Binding | null>(null);
  const [failure, setFailure] = useState<{
    identity: string;
    message: string;
    source: "load" | "action";
  } | null>(null);
  const [busyIdentity, setBusyIdentity] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const epoch = useRef(0);
  const operation = useRef<{ token: symbol; identity: string } | null>(null);
  const updated = useRef(onUpdated);
  useEffect(() => {
    updated.current = onUpdated;
  }, [onUpdated]);
  const current = binding?.identity === identity ? binding : null;

  useEffect(() => {
    const fence = ++epoch.current;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let unlisten: (() => void) | undefined;
    let previous: ComfyUiTestRun[] = [];
    const report = (error: unknown) => {
      if (epoch.current !== fence) return;
      const revoked = [401, 403, 404].includes(getApiErrorStatus(error) ?? 0);
      setFailure((previous) =>
        !revoked &&
        previous?.source === "action" &&
        previous.identity === identity
          ? previous
          : { identity, message: getApiErrorMessage(error), source: "load" },
      );
      if ([401, 403, 404].includes(getApiErrorStatus(error) ?? 0))
        setBinding(null);
    };
    void generationExecutionApi
      .scope()
      .then(async (scope) => {
        if (epoch.current !== fence) return;
        if (scope.userId !== userId)
          throw new Error("账号已变化，请重新读取试运行范围。");
        const key = comfyUiTestKey(scope, versionId);
        let pending: PendingComfyUiTest | null = null;
        let corrupt = false;
        try {
          pending = readPendingComfyUiTest(
            localStorage.getItem(key),
            scope,
            versionId,
          );
        } catch (error) {
          corrupt = true;
          report(error);
        }
        const poll = async () => {
          try {
            const runs = await comfyUiWorkflowApi.testHistory(versionId);
            if (epoch.current !== fence) return;
            try {
              pending = readPendingComfyUiTest(
                localStorage.getItem(key),
                scope,
                versionId,
              );
              corrupt = false;
            } catch (error) {
              pending = null;
              corrupt = true;
              report(error);
            }
            const finished = previous.some(
              (old) =>
                comfyUiTestActive(old) &&
                runs.some(
                  (run) => run.id === old.id && !comfyUiTestActive(run),
                ),
            );
            previous = runs;
            setBinding({ identity, scope, pending, corrupt, runs });
            if (!corrupt)
              setFailure((previous) =>
                previous?.source === "action" ? previous : null,
              );
            if (finished) await updated.current();
            if (epoch.current === fence && runs.some(comfyUiTestActive))
              timer = setTimeout(() => void poll(), 3000);
          } catch (error) {
            report(error);
            if (
              epoch.current === fence &&
              ![401, 403, 404].includes(getApiErrorStatus(error) ?? 0)
            )
              timer = setTimeout(() => void poll(), 5000);
          }
        };
        const storageChanged = (event: StorageEvent) => {
          if (event.key === key || event.key === null)
            setRefresh((value) => value + 1);
        };
        window.addEventListener("storage", storageChanged);
        unlisten = () => window.removeEventListener("storage", storageChanged);
        await poll();
      })
      .catch(report);
    return () => {
      epoch.current = fence + 1;
      clearTimeout(timer);
      unlisten?.();
    };
  }, [identity, userId, versionId, refresh]);

  const act = async (
    work: (scope: GenerationScope, valid: () => boolean) => Promise<void>,
  ) => {
    if (!current || current.corrupt || operation.current?.identity === identity)
      return false;
    const fence = epoch.current;
    const token = Symbol();
    operation.current = { token, identity };
    setBusyIdentity(identity);
    setFailure(null);
    const valid = () => epoch.current === fence;
    try {
      await work(current.scope, valid);
      if (valid()) {
        setFailure(null);
        setRefresh((value) => value + 1);
        await updated.current();
      }
      return true;
    } catch (error) {
      if (valid()) {
        setFailure({
          identity,
          message: getApiErrorMessage(error),
          source: "action",
        });
        if ([401, 403, 404].includes(getApiErrorStatus(error) ?? 0))
          setBinding(null);
      }
      return false;
    } finally {
      if (operation.current?.token === token) {
        operation.current = null;
        setBusyIdentity(null);
      }
    }
  };

  const submission = (
    action: "submit" | "confirm" | "withdraw",
    input?: Omit<ComfyUiTestRequest, "scope" | "requestId">,
  ) =>
    act(async (scope, valid) => {
      if (!navigator.locks)
        throw new Error(
          "当前浏览器不支持安全提交锁，请使用支持 Web Locks 的浏览器。",
        );
      const key = comfyUiTestKey(scope, versionId);
      await navigator.locks.request(
        key,
        { ifAvailable: true },
        async (lock) => {
          if (!lock)
            throw new Error("另一个页面正在处理这次试运行，请稍后刷新。");
          if (!valid()) return;
          let pending = readPendingComfyUiTest(
            localStorage.getItem(key),
            scope,
            versionId,
          );
          if (action === "submit") {
            if (pending)
              throw new Error("还有一条待确认请求，请先确认或撤回。");
            if (!input) throw new Error("缺少试运行输入。");
            pending = {
              schemaVersion: 1,
              createdAt: new Date().toISOString(),
              request: { ...input, scope, requestId: crypto.randomUUID() },
            };
            try {
              localStorage.setItem(key, JSON.stringify(pending));
            } catch {
              throw new Error(
                "无法保存待确认请求，尚未提交。图片较大时请先上传，再使用媒体引用。",
              );
            }
          }
          if (!pending)
            throw new Error("待确认请求已由其他页面处理，请刷新任务列表。");
          if (valid())
            setBinding((previous) =>
              previous?.identity === identity
                ? { ...previous, pending }
                : previous,
            );
          if (action === "withdraw")
            await comfyUiWorkflowApi.withdrawTestRequest(
              pending.request.requestId,
              scope,
            );
          else await comfyUiWorkflowApi.testVersion(pending.request);
          if (!valid()) return;
          const latest = readPendingComfyUiTest(
            localStorage.getItem(key),
            scope,
            versionId,
          );
          if (latest?.request.requestId !== pending.request.requestId)
            throw new Error("本地请求记录已变化，请刷新核对原任务。");
          localStorage.removeItem(key);
          setBinding((previous) =>
            previous?.identity === identity
              ? { ...previous, pending: null }
              : previous,
          );
        },
      );
    });

  const clearCorrupt = async () => {
    if (!current?.corrupt || !navigator.locks) return;
    const fence = epoch.current;
    const key = comfyUiTestKey(current.scope, versionId);
    await navigator.locks.request(key, { ifAvailable: true }, async (lock) => {
      if (!lock) throw new Error("其他页面正在使用该记录，请稍后再试。");
      if (epoch.current !== fence) return;
      let corrupt = false;
      try {
        readPendingComfyUiTest(
          localStorage.getItem(key),
          current.scope,
          versionId,
        );
      } catch {
        corrupt = true;
      }
      if (!corrupt) throw new Error("记录已经变化，请刷新后核对。");
      localStorage.removeItem(key);
      setRefresh((value) => value + 1);
    });
  };

  return {
    identity,
    runs: current?.runs ?? [],
    pending: current?.pending ?? null,
    corrupt: current?.corrupt ?? false,
    loading: !current,
    busy: busyIdentity === identity,
    error: failure?.identity === identity ? failure.message : null,
    reload: () => setRefresh((value) => value + 1),
    submit: (workflowHash: string, inputs: Record<string, unknown>) =>
      submission("submit", { versionId, workflowHash, inputs }),
    confirm: () => submission("confirm"),
    withdraw: () => submission("withdraw"),
    clearCorrupt,
    cancel: (run: ComfyUiTestRun) =>
      act(async (scope) => {
        await comfyUiWorkflowApi.cancelTest(run, scope);
      }),
    review: (run: ComfyUiTestRun, action: ComfyUiTestAction, note: string) =>
      act(async (scope) => {
        await comfyUiWorkflowApi.reviewTest(run, scope, action, note);
      }),
  };
}
