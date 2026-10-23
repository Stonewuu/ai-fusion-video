"use client";
import { useEffect, useState } from "react";
import { canvasApi, type CanvasResourceCollection } from "@/lib/api/canvas";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasResourceRef } from "@/lib/canvas/types";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import { useCanvas } from "../canvas-provider";

export function useCanvasCollection(ref: CanvasResourceRef, depth = 1) {
  const canvasId = useCanvas((state) => state.canvasId);
  const scope = useResourceChangeStore((state) => state.scope);
  const revision = useResourceChangeStore((state) => state.cursor);
  const [attempt, setAttempt] = useState(0);
  const { type, id } = ref;
  const identity = `${canvasId}:${type}:${id}:${depth}:${scope}`;
  const stamp = `${identity}:${revision}:${attempt}`;
  const [response, setResponse] = useState<{
    identity: string;
    stamp: string;
    result: CanvasResourceCollection | null;
    error: string | null;
  } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    void canvasApi
      .collection(canvasId, type, id, depth, controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) {
          setResponse({ identity, stamp, result: value, error: null });
        }
      })
      .catch((failure) => {
        if (!controller.signal.aborted) {
          setResponse({
            identity,
            stamp,
            result: null,
            error: getApiErrorMessage(failure),
          });
        }
      });
    return () => controller.abort();
  }, [canvasId, type, id, depth, identity, stamp]);
  return {
    result: response?.identity === identity ? response.result : null,
    error: response?.stamp === stamp ? response.error : null,
    loading: response?.stamp !== stamp,
    refresh: () => setAttempt((value) => value + 1),
  };
}
