import { canvasApi } from "@/lib/api/canvas";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasResource, CanvasResourceRef } from "./types";

export interface CanvasResourceState {
  data: CanvasResource | null;
  loading: boolean;
  error: string | null;
}
export const EMPTY_RESOURCE: CanvasResourceState = {
  data: null,
  loading: false,
  error: null,
};
type Entry = {
  ref: CanvasResourceRef;
  state: CanvasResourceState;
  listeners: Set<() => void>;
  abort: AbortController | null;
};

export function createCanvasResourceCache(canvasId: string) {
  let scopeGeneration = 0;
  const entries = new Map<string, Entry>();
  const key = (ref: CanvasResourceRef) => `${ref.type}:${ref.id}`;
  const entry = (ref: CanvasResourceRef) => {
    const id = key(ref);
    let current = entries.get(id);
    if (!current) {
      current = {
        ref,
        state: EMPTY_RESOURCE,
        listeners: new Set(),
        abort: null,
      };
      entries.set(id, current);
    }
    return current;
  };
  const update = (item: Entry, state: CanvasResourceState) => {
    item.state = state;
    item.listeners.forEach((listener) => listener());
  };
  const fetch = (item: Entry) => {
    item.abort?.abort();
    const controller = new AbortController();
    item.abort = controller;
    update(item, { ...item.state, loading: true, error: null });
    void canvasApi
      .resource(canvasId, item.ref.type, item.ref.id, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted)
          update(item, { data, loading: false, error: null });
      })
      .catch((error) => {
        if (!controller.signal.aborted)
          update(item, {
            data: null,
            loading: false,
            error: getApiErrorMessage(error),
          });
      });
  };
  return {
    generation: () => scopeGeneration,
    accept: (resource: CanvasResource, generation: number) => {
      if (generation !== scopeGeneration) return;
      const item = entry({
        type: resource.header.resourceType,
        id: resource.header.resourceId,
        version: resource.header.version,
      });
      if (
        item.state.data &&
        item.state.data.header.version > resource.header.version
      )
        return;
      item.abort?.abort();
      update(item, { data: resource, loading: false, error: null });
    },
    read: (ref: CanvasResourceRef | null) =>
      ref ? entry(ref).state : EMPTY_RESOURCE,
    subscribe: (ref: CanvasResourceRef | null, listener: () => void) => {
      if (!ref) return () => undefined;
      const item = entry(ref);
      item.listeners.add(listener);
      if (item.state === EMPTY_RESOURCE) fetch(item);
      return () => {
        item.listeners.delete(listener);
      };
    },
    invalidate: () => {
      entries.forEach((item) => {
        item.abort?.abort();
        if (item.listeners.size) fetch(item);
        else item.state = EMPTY_RESOURCE;
      });
    },
    clear: () => {
      scopeGeneration++;
      entries.forEach((item) => {
        item.abort?.abort();
        update(item, EMPTY_RESOURCE);
      });
    },
  };
}
export type CanvasResourceCache = ReturnType<typeof createCanvasResourceCache>;
