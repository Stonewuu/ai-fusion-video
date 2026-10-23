"use client";

import { useEffect, useRef, useState } from "react";
import {
  generationExecutionApi,
  type GenerationScope,
} from "@/lib/api/generation-execution";
import {
  mediaImportApi,
  type ImportPoint,
  type ImportSource,
  type ImportSummary,
  type MediaImport,
} from "@/lib/api/media-imports";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import {
  importRequestKey,
  parseImportRequest,
  type PendingImport,
} from "@/lib/canvas/import-state";
import { useCanvasStoreApi } from "../canvas-provider";

export function useMediaImportsController(userId: number) {
  const store = useCanvasStoreApi();
  const canvasId = store.getState().canvasId;
  const [scope, setScope] = useState<GenerationScope | null>(null);
  const [history, setHistory] = useState<ImportSummary[]>([]);
  const [active, setActive] = useState<ImportSummary[]>([]);
  const [job, setJob] = useState<MediaImport | null>(null);
  const [pending, setPending] = useState<PendingImport | null>(null);
  const [corrupt, setCorrupt] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [mediaRevision, setMediaRevision] = useState(0);
  const importedCounts = useRef(new Map<string, number>());
  const errorSource = useRef<"action" | "load" | null>(null);
  const alive = useRef(false);
  const operation = useRef(false);
  const selected = useRef<string | null>(null);
  const sequence = useRef(0);
  const placementFailures = useRef(new Map<string, number>());
  const [placementError, setPlacementError] = useState<string | null>(null);
  const reload = () => setRefresh((v) => v + 1);

  useEffect(() => {
    alive.current = true;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      const read = ++sequence.current;
      const valid = () =>
        !controller.signal.aborted && read === sequence.current;
      try {
        const currentScope = await generationExecutionApi.scope();
        if (!valid()) return;
        if (currentScope.userId !== userId)
          throw new Error("登录账号已经变化，请重新打开画布");
        const raw = localStorage.getItem(
          importRequestKey(canvasId, currentScope),
        );
        let saved: PendingImport | null = null;
        let broken = false;
        try {
          if (raw) saved = parseImportRequest(raw, canvasId, currentScope);
        } catch (failure) {
          broken = true;
          errorSource.current = "load";
          setError(getApiErrorMessage(failure));
        }
        const [entries, ongoing] = await Promise.all([
          mediaImportApi.list(canvasId, page, controller.signal),
          mediaImportApi.active(canvasId, controller.signal),
        ]);
        const chosen = selected.current;
        let detail = chosen
          ? await mediaImportApi.get(chosen, controller.signal)
          : null;
        if (!valid()) return;
        setScope(currentScope);
        setHistory(entries.items);
        setHasMore(entries.hasMore);
        setActive(ongoing);
        setPending(saved);
        setCorrupt(broken);
        if (!broken && errorSource.current === "load") {
          setError(null);
          errorSource.current = null;
        }
        let mediaChanged = false;
        for (const entry of [...entries.items, ...ongoing]) {
          const previous = importedCounts.current.get(entry.id) ?? 0;
          if (entry.succeeded > previous) {
            importedCounts.current.set(entry.id, entry.succeeded);
            mediaChanged = true;
          }
        }
        if (mediaChanged) setMediaRevision((v) => v + 1);
        const canvas = store.getState();
        if (
          !operation.current &&
          canvas.snapshot?.permissions.canEdit &&
          !canvas.snapshot.document.archived &&
          !canvas.pending &&
          !canvas.saving
        ) {
          for (const summary of ongoing) {
            if (!valid() || operation.current) break;
            if (
              !summary.readyToPlace ||
              !summary.automaticPlacement ||
              summary.cancelRequested ||
              placementFailures.current.get(summary.id) === summary.revision
            )
              continue;
            try {
              const placed = await mediaImportApi.place(
                summary.id,
                currentScope,
                true,
              );
              if (!valid()) return;
              if (placed.command)
                store.getState().rememberCommand(placed.command);
              placementFailures.current.delete(summary.id);
              if (detail?.id === placed.job.id) detail = placed.job;
            } catch (failure) {
              if (!valid()) return;
              placementFailures.current.set(summary.id, summary.revision);
              setPlacementError(
                `素材已保留，放入画布尚未完成：${getApiErrorMessage(failure)}`,
              );
            }
          }
          if (!placementFailures.current.size) setPlacementError(null);
        }
        if (!valid()) return;
        setJob((old) =>
          old && detail?.id === old.id && old.revision > detail.revision
            ? old
            : detail,
        );
        timer = setTimeout(
          () => void load(),
          ongoing.length || saved ? 2500 : 8000,
        );
      } catch (failure) {
        if (!valid()) return;
        errorSource.current = "load";
        setError(getApiErrorMessage(failure));
        if ([401, 403, 404].includes(getApiErrorStatus(failure) ?? 0)) {
          setScope(null);
          setJob(null);
          setHistory([]);
          setActive([]);
          setPending(null);
        } else timer = setTimeout(() => void load(), 5000);
      }
    };
    const changed = () => {
      clearTimeout(timer);
      void load();
    };
    void load();
    window.addEventListener("storage", changed);
    return () => {
      alive.current = false;
      controller.abort();
      clearTimeout(timer);
      window.removeEventListener("storage", changed);
    };
  }, [canvasId, userId, page, refresh, store]);

  const act = async (
    work: (scope: GenerationScope) => Promise<void>,
    allowCorrupt = false,
  ) => {
    if (!scope || operation.current || (corrupt && !allowCorrupt)) return false;
    operation.current = true;
    setBusy(true);
    setError(null);
    errorSource.current = "action";
    ++sequence.current;
    try {
      await work(scope);
      if (alive.current) {
        errorSource.current = null;
        reload();
      }
      return true;
    } catch (failure) {
      if (alive.current) setError(getApiErrorMessage(failure));
      return false;
    } finally {
      operation.current = false;
      if (alive.current) {
        setBusy(false);
        reload();
      }
    }
  };
  const submit = (
    mode: "start" | "resume" | "withdraw",
    sources?: ImportSource[],
    position: ImportPoint | null = null,
  ) =>
    act(async (scope) => {
      if (!navigator.locks) throw new Error("当前浏览器不支持安全提交锁");
      const key = importRequestKey(canvasId, scope);
      await navigator.locks.request(
        key,
        { ifAvailable: true },
        async (lock) => {
          if (!lock) throw new Error("另一个页面正在处理此画布的导入请求");
          if (!alive.current) return;
          const raw = localStorage.getItem(key);
          let saved = raw ? parseImportRequest(raw, canvasId, scope) : null;
          if (mode === "start") {
            if (saved) throw new Error("请先确认或撤回上一条导入请求");
            if (!sources?.length || sources.length > 50)
              throw new Error("每批请选择 1–50 项素材");
            saved = {
              version: 1,
              canvasId,
              createdAt: new Date().toISOString(),
              request: {
                requestId: crypto.randomUUID(),
                scope,
                sources,
                position,
              },
            };
            parseImportRequest(JSON.stringify(saved), canvasId, scope);
            localStorage.setItem(key, JSON.stringify(saved));
          }
          if (!saved) throw new Error("此请求已由其他页面处理，请刷新记录");
          setPending(saved);
          const result =
            mode === "withdraw"
              ? (
                  await mediaImportApi.withdraw(
                    canvasId,
                    saved.request.requestId,
                    scope,
                  )
                ).job
              : await mediaImportApi.submit(canvasId, saved.request);
          const latest = localStorage.getItem(key);
          if (
            latest &&
            parseImportRequest(latest, canvasId, scope).request.requestId ===
              saved.request.requestId
          )
            localStorage.removeItem(key);
          if (!alive.current) return;
          setPending(null);
          setPage(0);
          if (result) {
            selected.current = result.id;
            setJob(result);
          }
        },
      );
    });
  return {
    mediaRevision,
    history,
    active,
    job,
    pending,
    corrupt,
    error,
    placementError,
    busy,
    page,
    hasMore,
    ready: !!scope && !corrupt && !busy && !pending,
    reload,
    setPage,
    choose: (id: string | null) => {
      selected.current = id;
      setJob(null);
      reload();
    },
    start: (sources: ImportSource[], point: ImportPoint | null) =>
      submit("start", sources, point),
    resume: () => submit("resume"),
    withdraw: () => submit("withdraw"),
    cancel: (current: MediaImport) =>
      act(async (scope) => {
        const result = await mediaImportApi.cancel(current.id, scope);
        if (alive.current) setJob(result);
      }),
    retry: (current: MediaImport, indices: number[]) =>
      act(async (scope) => {
        const result = await mediaImportApi.retry(current, indices, scope);
        if (alive.current) setJob(result);
      }),
    place: (current: MediaImport, point: ImportPoint | null) =>
      act(async (scope) => {
        const result = await mediaImportApi.place(
          current.id,
          scope,
          false,
          point,
        );
        placementFailures.current.delete(current.id);
        if (alive.current) {
          if (result.command) store.getState().rememberCommand(result.command);
          setJob(result.job);
        }
      }),
    clearCorrupt: () =>
      act(async (scope) => {
        if (!navigator.locks) throw new Error("当前浏览器不支持安全提交锁");
        const key = importRequestKey(canvasId, scope);
        await navigator.locks.request(
          key,
          { ifAvailable: true },
          async (lock) => {
            if (!lock) throw new Error("另一页面正在操作导入请求");
            const raw = localStorage.getItem(key);
            if (!raw) return;
            let invalid = false;
            try {
              parseImportRequest(raw, canvasId, scope);
            } catch {
              invalid = true;
            }
            if (!invalid) throw new Error("请求已恢复正常，请重新核对记录");
            localStorage.removeItem(key);
            setCorrupt(false);
            setPending(null);
          },
        );
      }, true),
  };
}
