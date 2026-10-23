"use client";
import { useState, type SetStateAction } from "react";
import { useAuthStore } from "@/lib/store/auth-store";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasNode } from "@/lib/canvas/types";
import {
  changedNodeFields,
  isCanvasConfig,
  type CanvasNodeDraft,
} from "@/lib/canvas/node-draft";
import { useCanvas } from "../canvas-provider";

export function useCanvasNodeDraft(node: CanvasNode) {
  const userId = useAuthStore((state) => state.user?.id);
  const canvasId = useCanvas((state) => state.canvasId);
  const key = `canvas-node-draft:${userId}:${canvasId}:${node.id}`;
  const [initial] = useState(() => {
    try {
      const raw = sessionStorage.getItem(key);
      if (!raw) return { draft: null, error: null };
      if (raw.length > 2 * 1024 * 1024)
        throw new Error("节点草稿超过允许大小，请清理后重新编辑");
      const value = JSON.parse(raw) as CanvasNodeDraft;
      if (
        value?.base?.id !== node.id ||
        value.base.kind !== node.kind ||
        typeof value.base.title !== "string" ||
        !isCanvasConfig(value.base.config) ||
        !Number.isSafeInteger(value.base.configVersion) ||
        typeof value.title !== "string" ||
        !isCanvasConfig(value.config)
      )
        throw new Error("节点草稿格式无效，请清理后重新编辑");
      return { draft: value, error: null };
    } catch (failure) {
      return { draft: null, error: getApiErrorMessage(failure) };
    }
  });
  const [draft, update] = useState<CanvasNodeDraft | null>(initial.draft);
  const [storageError, setStorageError] = useState(initial.error);
  const [invalid, setInvalid] = useState(!!initial.error);
  const setDraft = (action: SetStateAction<CanvasNodeDraft | null>) => {
    const value = typeof action === "function" ? action(draft) : action;
    const next = value && changedNodeFields(value).length ? value : null;
    try {
      if (next) sessionStorage.setItem(key, JSON.stringify(next));
      else sessionStorage.removeItem(key);
      setStorageError(null);
      setInvalid(false);
    } catch (failure) {
      setStorageError(`草稿未能保存在此浏览器：${getApiErrorMessage(failure)}`);
    }
    update(next);
  };
  return { draft, setDraft, storageError, invalid, draftKey: key };
}
