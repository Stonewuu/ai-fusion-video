"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import { useStore } from "zustand";
import {
  createCanvasStore,
  type CanvasStore,
  type CanvasState,
} from "@/lib/store/canvas-store";
import {
  createCanvasResourceCache,
  EMPTY_RESOURCE,
  type CanvasResourceCache,
} from "@/lib/canvas/resource-cache";
import { CanvasStreamError, streamCanvas } from "@/lib/canvas/event-stream";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import type { CanvasResourceRef } from "@/lib/canvas/types";
import { useCanvasSaveRecovery } from "./use-save-recovery";
import { subscribeAgentConfiguration } from "@/lib/agent-configuration-events";
import { useAuthStore } from "@/lib/store/auth-store";

const Context = createContext<{
  store: CanvasStore;
  resources: CanvasResourceCache;
} | null>(null);

function delay(milliseconds: number, signal: AbortSignal) {
  return new Promise<void>((resolve) => {
    const abort = () => {
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, milliseconds);
    signal.addEventListener("abort", abort, { once: true });
  });
}

export function CanvasProvider({
  canvasId,
  children,
}: {
  canvasId: string;
  children: ReactNode;
}) {
  const [value] = useState(() => ({
    store: createCanvasStore(canvasId),
    resources: createCanvasResourceCache(canvasId),
  }));
  const resourceRevision = useResourceChangeStore((state) => state.cursor);
  const userId = useAuthStore((state) => state.user?.id);
  useEffect(() => {
    if (!userId) return;
    return subscribeAgentConfiguration(userId, () => {
      void value.store.getState().refreshCatalog().catch((error) => value.store.setState({ error: getApiErrorMessage(error) }));
    });
  }, [userId, value]);
  useCanvasSaveRecovery(value.store);
  const resourceScope = useResourceChangeStore((state) => state.scope);
  useEffect(() => {
    value.resources.invalidate();
  }, [value, resourceRevision]);
  useEffect(() => {
    const controller = new AbortController();
    value.resources.clear();
    value.store.getState().reauthorize();
    const { signal } = controller;
    let attempts = 0;
    void (async () => {
      while (!signal.aborted) {
        try {
          await value.store.getState().load(signal);
          if (signal.aborted) return;
          const cursor = value.store.getState().snapshot!.lastSequence;
          await streamCanvas(
            canvasId,
            cursor,
            signal,
            value.store.getState().receive,
            () => {
              attempts = 0;
              value.store.setState({ sync: "connected", syncError: null });
            },
          );
        } catch (error) {
          if (signal.aborted) return;
          const denied =
            [403, 404].includes(
              getApiErrorStatus(error) ??
                (error instanceof CanvasStreamError ? (error.status ?? 0) : 0),
            ) ||
            (error instanceof CanvasStreamError &&
              ["ACCESS_DENIED", "RESOURCE_DELETED"].includes(error.code));
          const message = getApiErrorMessage(error);
          if (denied) {
            value.store.setState({
              snapshot: null,
              catalog: null,
              pending: null,
              selection: [],
              edgeSelection: [],
              sync: "denied",
              error: message,
              loading: false,
            });
            value.resources.clear();
            return;
          }
          value.store.setState({
            loading: false,
            sync: "retrying",
            syncError: message,
            error: value.store.getState().snapshot ? null : message,
          });
          await delay(Math.min(15000, 1000 * 2 ** attempts++), signal);
        }
      }
    })();
    return () => {
      controller.abort();
      value.resources.clear();
    };
  }, [canvasId, value, resourceScope]);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export function useCanvasStoreApi() {
  const context = useContext(Context);
  if (!context) throw new Error("画布组件必须位于 CanvasProvider 内");
  return context.store;
}
export function useCanvasResourceCache() {
  const context = useContext(Context);
  if (!context) throw new Error("业务读取必须位于画布工作区内");
  return context.resources;
}
export function useCanvas<T>(selector: (state: CanvasState) => T): T {
  return useStore(useCanvasStoreApi(), selector);
}
export function useCanvasResource(ref: CanvasResourceRef | null) {
  const context = useContext(Context);
  if (!context) throw new Error("业务读取必须位于画布工作区内");
  const { resources } = context;
  const type = ref?.type;
  const id = ref?.id;
  const version = ref?.version;
  const stableRef = useCallback(
    () =>
      type && id !== undefined && version !== undefined
        ? { type, id, version }
        : null,
    [type, id, version],
  );
  return useSyncExternalStore(
    useCallback(
      (listener) => resources.subscribe(stableRef(), listener),
      [resources, stableRef],
    ),
    useCallback(() => resources.read(stableRef()), [resources, stableRef]),
    () => EMPTY_RESOURCE,
  );
}
