"use client";

import { useEffect, useRef, useState } from "react";
import {
  canvasRunApi,
  type CanvasRun,
  type CanvasRunPlan,
  type CanvasRunMode,
  type CanvasRunBudget,
  type CanvasNodeRun,
  type CanvasRunSummary,
  type CanvasNodeActivity,
} from "@/lib/api/canvas-runs";
import {
  generationExecutionApi,
  type GenerationScope,
} from "@/lib/api/generation-execution";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import {
  defaultRunBudget,
  parseRunRequest,
  runFinished,
  runRequestKey,
  type PendingCanvasRun,
} from "@/lib/canvas/run-state";
import { useCanvasStoreApi } from "../canvas-provider";
import { prepareCanvasNodeRun } from "@/lib/canvas/prepare-node-run";
import type { CanvasNodeStartResult } from "@/lib/canvas/node-execution";

export function useCanvasRunsController(userId: number) {
  const store = useCanvasStoreApi();
  const canvasId = store.getState().canvasId;
  const [scope, setScope] = useState<GenerationScope | null>(null);
  const [plan, setPlan] = useState<CanvasRunPlan | null>(null);
  const [run, setRun] = useState<CanvasRun | null>(null);
  const [activities, setActivities] = useState<CanvasNodeActivity[]>([]);
  const [history, setHistory] = useState<CanvasRunSummary[]>([]);
  const [pending, setPending] = useState<PendingCanvasRun | null>(null);
  const [corrupt, setCorrupt] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [historyPage, setHistoryPage] = useState(0);
  const alive = useRef(false);
  const operation = useRef(false);
  const selected = useRef<string | null>(null);
  const readSequence = useRef(0);
  const failureSource = useRef<"action" | "load" | null>(null);
  const reload = () => setRefresh((value) => value + 1);

  useEffect(() => {
    alive.current = true;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const report = (failure: unknown) => {
      if (controller.signal.aborted) return;
      const denied = [401, 403, 404].includes(getApiErrorStatus(failure) ?? 0);
      if (denied || failureSource.current !== "action") {
        setError(getApiErrorMessage(failure));
        failureSource.current = "load";
      }
      if (denied) {
        setScope(null);
        setRun(null);
        setActivities([]);
        setHistory([]);
        setPending(null);
        setPlan(null);
      } else timer = setTimeout(() => void load(), 5000);
    };
    const load = async () => {
      const sequence = ++readSequence.current;
      try {
        const currentScope = await generationExecutionApi.scope();
        if (controller.signal.aborted || sequence !== readSequence.current)
          return;
        if (currentScope.userId !== userId)
          throw new Error("登录账号已经变化，请重新打开画布");
        const key = runRequestKey(canvasId, currentScope);
        let request: PendingCanvasRun | null = null;
        let broken = false;
        try {
          const raw = localStorage.getItem(key);
          if (raw) request = parseRunRequest(raw, canvasId, currentScope);
        } catch (failure) {
          broken = true;
          setError(getApiErrorMessage(failure));
        }
        const [entries, nodeActivities] = await Promise.all([
          canvasRunApi.list(canvasId, historyPage, controller.signal),
          canvasRunApi.nodeActivities(canvasId, controller.signal),
        ]);
        const chosen = selected.current ?? entries[0]?.id;
        const current = chosen
          ? await canvasRunApi.get(chosen, controller.signal)
          : null;
        if (controller.signal.aborted || sequence !== readSequence.current)
          return;
        setScope(currentScope);
        setHistory(entries);
        setActivities(nodeActivities);
        setPending(request);
        setCorrupt(broken);
        setRun((old) =>
          old?.id === current?.id &&
          old &&
          current &&
          old.revision > current.revision
            ? old
            : current,
        );
        if (!broken && failureSource.current === "load") {
          setError(null);
          failureSource.current = null;
        }
        timer = setTimeout(
          () => void load(),
          nodeActivities.some((item) => !runFinished(item.node.status)) || current && !runFinished(current.status) ? 2500 : 8000,
        );
      } catch (failure) {
        report(failure);
      }
    };
    void load();
    const changed = () => {
      clearTimeout(timer);
      void load();
    };
    window.addEventListener("storage", changed);
    return () => {
      alive.current = false;
      controller.abort();
      clearTimeout(timer);
      window.removeEventListener("storage", changed);
    };
  }, [canvasId, userId, historyPage, refresh]);

  const act = async (
    work: (scope: GenerationScope) => Promise<void>,
    allowCorrupt = false,
    onFailure?: (message: string) => void,
  ) => {
    if (!scope || operation.current || (corrupt && !allowCorrupt)) {
      onFailure?.(!scope ? "运行服务尚未准备好，请稍后再试。" : operation.current ? "正在提交另一项操作，请稍后再试。" : "上次提交记录需要处理，请打开运行面板。");
      return false;
    }
    operation.current = true;
    ++readSequence.current;
    setBusy(true);
    setError(null);
    failureSource.current = "action";
    try {
      await work(scope);
      if (alive.current) {
        setError(null);
        failureSource.current = null;
      }
      return true;
    } catch (failure) {
      onFailure?.(getApiErrorMessage(failure));
      if (alive.current) {
        setError(getApiErrorMessage(failure));
        if ([401, 403, 404].includes(getApiErrorStatus(failure) ?? 0)) {
          setRun(null);
          setActivities([]);
          setHistory([]);
          setPlan(null);
          setScope(null);
        }
      }
      return false;
    } finally {
      operation.current = false;
      if (alive.current) { setBusy(false); reload(); }
    }
  };
  const preview = (
    mode: CanvasRunMode,
    nodeIds: string[],
    reuse: Record<string, string> = {},
    budget: CanvasRunBudget = defaultRunBudget,
  ) =>
    act(async (scope) => {
      const current = store.getState();
      if (
        !current.snapshot ||
        current.pending ||
        current.saving ||
        current.sync !== "connected"
      )
        throw new Error("请先完成保存并恢复同步，再预览运行计划");
      setPlan(null);
      const value = await canvasRunApi.plan(canvasId, {
        expectedRevision: current.snapshot.document.revision,
        mode,
        nodeIds,
        reuse,
        budget,
        scope,
      });
      if (alive.current) setPlan(value);
    });
  const submit = (
    action: "start" | "confirm" | "withdraw",
    preview?: CanvasRunPlan,
    prepare?: (scope: GenerationScope) => Promise<CanvasRunPlan | null>,
    onFailure?: (message: string) => void,
  ) =>
    act(async (scope) => {
      if (!navigator.locks)
        throw new Error(
          "当前浏览器不支持安全提交锁，请使用支持 Web Locks 的浏览器",
        );
      const key = runRequestKey(canvasId, scope);
      await navigator.locks.request(
        key,
        { ifAvailable: true },
        async (lock) => {
          if (!lock)
            throw new Error("另一个页面正在处理此画布的运行请求，请稍后刷新");
          if (!alive.current) return;
          const raw = localStorage.getItem(key);
          let request = raw ? parseRunRequest(raw, canvasId, scope) : null;
          if (action === "start") {
            if (request) throw new Error("还有一条待确认请求，请先确认或撤回");
            if (prepare) {
              const prepared = await prepare(scope);
              if (!prepared) return;
              preview = prepared;
            }
            if (!preview || preview.issues.length)
              throw new Error("请先完成运行前检查");
            const document = store.getState().snapshot?.document;
            if (
              !document ||
              document.revision !== preview.documentRevision ||
              Date.parse(preview.expiresAt) <= Date.now()
            )
              throw new Error("画布或计划已变化，请重新预览");
            request = {
              version: 1,
              canvasId,
              title: document.title,
              cost: preview.cost,
              createdAt: new Date().toISOString(),
              request: {
                requestId: crypto.randomUUID(),
                planId: preview.id,
                planHash: preview.hash,
                scope,
              },
            };
            localStorage.setItem(key, JSON.stringify(request));
          }
          if (!request)
            throw new Error("原请求已由其他页面处理，请刷新运行记录");
          setPending(request);
          let result: CanvasRun | null;
          try {
            result = action === "withdraw" ? await canvasRunApi.withdraw(
                  canvasId,
                  request.request.requestId,
                  scope,
                )
              : await canvasRunApi.start(canvasId, request.request);
          } catch (failure) {
            if (action === "start" && [400, 401, 403, 404, 409, 410, 422].includes(getApiErrorStatus(failure) ?? 0)) {
              const saved = localStorage.getItem(key);
              if (saved && parseRunRequest(saved, canvasId, scope).request.requestId === request.request.requestId) localStorage.removeItem(key);
              if (alive.current) setPending(null);
            }
            throw failure;
          }
          if (!alive.current) return;
          const latest = localStorage.getItem(key);
          if (
            !latest ||
            parseRunRequest(latest, canvasId, scope).request.requestId !==
              request.request.requestId
          )
            throw new Error("本地请求记录发生变化，请核对原运行");
          localStorage.removeItem(key);
          setPending(null);
          setPlan(null);
          if (result) {
            selected.current = result.id;
            setRun(result);
            setActivities((current) => {
              const changed = new Set(result.nodes.filter((node) => node.parentNodeRunId === null && node.action !== "REUSE").map((node) => node.nodeId));
              return [...current.filter((item) => !changed.has(item.node.nodeId)), ...result.nodes.filter((node) => node.parentNodeRunId === null && node.action !== "REUSE").map((node) => ({
                runId: result.id, runStatus: result.status, canControl: result.canControl, node,
                children: result.nodes.filter((child) => child.parentNodeRunId === node.id),
              }))];
            });
          }
        },
      );
    }, false, onFailure);
  const startNode = async (nodeId: string): Promise<CanvasNodeStartResult> => {
    let error: string | null = null;
    let review: CanvasNodeStartResult["review"];
    const accepted = await submit("start", undefined, async (currentScope) => {
      const { plan: prepared, reuse } = await prepareCanvasNodeRun(store, userId, currentScope, nodeId);
      if (prepared.nodes.some((node) => node.action !== "REUSE" && node.externalOperation && node.externalOperation.effect !== "READ_ONLY")) {
        setPlan(prepared);
        review = { mode: "DEPENDENCIES", nodeIds: [nodeId], reuse };
        return null;
      }
      return prepared;
    }, (message) => { error = message; });
    return { started: accepted && !review, error, review };
  };
  return {
    scope,
    plan,
    run,
    activities,
    history,
    historyPage,
    pending,
    corrupt,
    error,
    busy,
    loading: !scope,
    preview,
    invalidatePlan: () => setPlan(null),
    start: () => submit("start", plan ?? undefined),
    startNode,
    confirm: () => submit("confirm"),
    withdraw: () => submit("withdraw"),
    cancel: () =>
      act(async (scope) => {
        if (run) {
          const result = await canvasRunApi.cancel(run.id, scope);
          if (alive.current) setRun(result);
        }
      }),
    select: (node: CanvasNodeRun, outputId: string) =>
      act(async (scope) => {
        if (run) {
          const result = await canvasRunApi.select(
            run.id,
            node,
            outputId,
            scope,
          );
          if (alive.current) setRun(result);
        }
      }),
    chooseRun: (id: string) => {
      selected.current = id;
      setRun(null);
      reload();
    },
    page: (next: number) => {
      setHistoryPage(next);
    },
    reload,
    clearCorrupt: () =>
      act(async (scope) => {
        if (!navigator.locks) throw new Error("浏览器不支持安全提交锁");
        const key = runRequestKey(canvasId, scope);
        await navigator.locks.request(
          key,
          { ifAvailable: true },
          async (lock) => {
            if (!lock) throw new Error("另一页面正在操作此请求，请稍后再试");
            const raw = localStorage.getItem(key);
            if (!raw) return;
            let invalid = false;
            try {
              parseRunRequest(raw, canvasId, scope);
            } catch {
              invalid = true;
            }
            if (!invalid) throw new Error("请求内容已恢复正常，请刷新核对");
            localStorage.removeItem(key);
            setCorrupt(false);
            setPending(null);
          },
        );
      }, true),
  };
}
