"use client";
import { useEffect, useState } from "react";
import {
  loadCanvasLibrary,
  type CanvasLibraryEntry,
} from "@/lib/canvas/resource-library";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCanvas } from "../canvas-provider";

export function useCanvasLibrary(enabled = true) {
  const projectId = useCanvas((state) => state.snapshot!.document.projectId);
  const revision = useResourceChangeStore((state) => state.cursor);
  const scope = useResourceChangeStore((state) => state.scope);
  const [attempt, setAttempt] = useState(0);
  const key = `${projectId}:${scope}:${revision}:${attempt}`;
  const [result, setResult] = useState<{
    key: string;
    entries: CanvasLibraryEntry[];
    error: string | null;
  } | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let active = true;
    void loadCanvasLibrary(projectId)
      .then((entries) => {
        if (active) setResult({ key, entries, error: null });
      })
      .catch((failure) => {
        if (active)
          setResult({ key, entries: [], error: getApiErrorMessage(failure) });
      });
    return () => {
      active = false;
    };
  }, [projectId, key, enabled]);
  return {
    entries: result?.key === key ? result.entries : [],
    error: result?.key === key ? result.error : null,
    loading: enabled && result?.key !== key,
    refresh: () => setAttempt((value) => value + 1),
  };
}
