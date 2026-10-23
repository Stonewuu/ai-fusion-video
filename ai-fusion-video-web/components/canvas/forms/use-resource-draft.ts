"use client";

import { useRef, useState } from "react";
import { useCanvasAssistantDraft } from "../agent/use-assistant-draft";
import { useAuthStore } from "@/lib/store/auth-store";
import { canvasApi } from "@/lib/api/canvas";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import {
  acknowledgeResourceDraft,
  normalizeResourcePatch,
  type CanvasResourceDraft,
} from "@/lib/canvas/resource-draft";
import type { CanvasResource } from "@/lib/canvas/types";
import {
  useCanvas,
  useCanvasResourceCache,
  useCanvasStoreApi,
} from "../canvas-provider";

interface SavedDraft {
  draft: CanvasResourceDraft;
  pending: CanvasResourceDraft | null;
}
function readDraft(key: string, resource: CanvasResource): SavedDraft {
  const stored = sessionStorage.getItem(key);
  if (!stored)
    return {
      draft: { baseline: resource, patch: {}, requestId: crypto.randomUUID() },
      pending: null,
    };
  const value = JSON.parse(stored) as SavedDraft;
  if (
    value?.draft?.baseline?.header?.resourceId !== resource.header.resourceId ||
    value.draft.baseline.header.resourceType !== resource.header.resourceType ||
    !Number.isSafeInteger(value.draft.baseline.header.version) ||
    typeof value.draft.patch !== "object" ||
    !value.draft.patch ||
    typeof value.draft.requestId !== "string"
  ) {
    throw new Error("本地草稿格式不正确，请先清理该对象的草稿");
  }
  return value;
}

export function useCanvasResourceDraft(resource: CanvasResource) {
  const canvasId = useCanvas((state) => state.canvasId);
  const userId = useAuthStore((state) => state.user?.id);
  const store = useCanvasStoreApi();
  const cache = useCanvasResourceCache();
  const key = `canvas-draft:${userId}:${canvasId}:${resource.header.resourceType}:${resource.header.resourceId}`;
  const [initial] = useState(() => {
    try {
      return { saved: readDraft(key, resource), error: null };
    } catch (error) {
      return { saved: null, error: getApiErrorMessage(error) };
    }
  });
  const [saved, setSaved] = useState<SavedDraft | null>(initial.saved);
  const savedRef = useRef(initial.saved);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(initial.error);
  const [storageError, setStorageError] = useState<string | null>(null);
  const persist = (next: SavedDraft) => {
    try {
      if (!Object.keys(next.draft.patch).length && !next.pending)
        sessionStorage.removeItem(key);
      else sessionStorage.setItem(key, JSON.stringify(next));
      setStorageError(null);
    } catch (failure) {
      setStorageError(`草稿未能保存到此浏览器：${getApiErrorMessage(failure)}`);
    }
    savedRef.current = next;
    setSaved(next);
  };
  const change = (field: string, value: unknown) => {
    const current = savedRef.current;
    if (!current) return;
    const baseline =
      !current.pending &&
      !Object.keys(current.draft.patch).length &&
      resource.header.version >= current.draft.baseline.header.version
        ? resource
        : current.draft.baseline;
    const patch = { ...current.draft.patch };
    if (JSON.stringify(value) === JSON.stringify(baseline.data[field]))
      delete patch[field];
    else patch[field] = value;
    persist({
      ...current,
      draft: { baseline, patch, requestId: crypto.randomUUID() },
    });
  };
  const reset = () => {
    if (saving || saved?.pending) return;
    const next = {
      draft: { baseline: resource, patch: {}, requestId: crypto.randomUUID() },
      pending: null,
    };
    persist(next);
    setError(null);
  };
  const rebase = (serverFields: string[]) => {
    if (saving || !saved || saved.pending) return;
    const patch = { ...saved.draft.patch };
    serverFields.forEach((field) => delete patch[field]);
    persist({
      draft: { baseline: resource, patch, requestId: crypto.randomUUID() },
      pending: null,
    });
    setError(null);
  };
  const save = async () => {
    const saved = savedRef.current;
    if (!saved || saving) return;
    const generation = cache.generation();
    const submitted = saved.pending ?? saved.draft;
    const type = submitted.baseline.header.resourceType;
    let patch: Record<string, unknown>;
    try {
      patch = normalizeResourcePatch(type, submitted.patch);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
      return;
    }
    if (!Object.keys(patch).length) return;
    setSaving(true);
    setError(null);
    persist({ ...saved, pending: submitted });
    try {
      const receipt = await canvasApi.patchResource(canvasId, {
        requestId: submitted.requestId,
        resourceType: type,
        resourceId: submitted.baseline.header.resourceId,
        expectedVersion: submitted.baseline.header.version,
        patch,
        ...(type === "assetItem"
          ? { expectedParentVersion: submitted.baseline.data.assetVersion }
          : {}),
      });
      if (generation !== cache.generation())
        throw new Error("访问范围已变化，请在重新打开内容后核对刚才的保存结果");
      if (!Number.isSafeInteger(receipt.result.version))
        throw new Error("业务保存未返回有效版本，请重试确认提交结果");
      const acknowledged: CanvasResource = {
        ...submitted.baseline,
        header: {
          ...submitted.baseline.header,
          version: receipt.result.version as number,
        },
        data: receipt.result,
      };
      const current = savedRef.current;
      if (current)
        persist({
          draft: acknowledgeResourceDraft(
            current.draft,
            submitted,
            acknowledged,
          ),
          pending: null,
        });
      cache.accept(acknowledged, generation);
      await store.getState().load();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
      const status = getApiErrorStatus(failure);
      if (
        status !== null &&
        status >= 400 &&
        status < 500 &&
        savedRef.current
      ) {
        persist({ ...savedRef.current, pending: null });
        cache.invalidate();
      }
    } finally {
      setSaving(false);
    }
  };
  const draft = saved
    ? !saved.pending &&
      !Object.keys(saved.draft.patch).length &&
      resource.header.version >= saved.draft.baseline.header.version
      ? { ...saved.draft, baseline: resource }
      : saved.draft
    : null;
  useCanvasAssistantDraft(key, save);
  return {
    draft,
    pending: saved?.pending ?? null,
    saving,
    error,
    storageError,
    change,
    reset,
    rebase,
    save,
  };
}
