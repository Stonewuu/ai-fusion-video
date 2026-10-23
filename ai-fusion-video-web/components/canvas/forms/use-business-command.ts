"use client";
import { useState } from "react";
import { canvasApi, type ResourceReceipt } from "@/lib/api/canvas";
import { useAuthStore } from "@/lib/store/auth-store";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCanvas, useCanvasResourceCache } from "../canvas-provider";

type Operation = Parameters<typeof canvasApi.resourceCommand>[1];
type Pending = { operation: Operation; command: Record<string, unknown> };
const operations: Operation[] = [
  "create",
  "delete",
  "script-source",
  "script-scenes",
  "asset-batch",
  "storyboard-children",
  "storyboard-batch",
];

export function useCanvasBusinessCommand(
  scope: string,
  onCommitted: (receipt: ResourceReceipt) => void,
) {
  const canvasId = useCanvas((state) => state.canvasId);
  const userId = useAuthStore((state) => state.user?.id);
  const cache = useCanvasResourceCache();
  const key = `canvas-business-command:${userId}:${canvasId}:${scope}`;
  const [initial] = useState(() => {
    try {
      const raw = sessionStorage.getItem(key);
      if (!raw) return { pending: null };
      const parsed = JSON.parse(raw);
      if (
        !operations.includes(parsed.operation) ||
        !parsed.command ||
        typeof parsed.command.requestId !== "string"
      )
        throw new Error();
      return { pending: parsed as Pending };
    } catch {
      return {
        pending: null,
        error: "上次操作记录无法读取，请清理后核对历史记录。",
      };
    }
  });
  const [pending, setPending] = useState<Pending | null>(initial.pending);
  const [error, setError] = useState<string | null>(initial.error ?? null);
  const [busy, setBusy] = useState(false);
  const [invalid, setInvalid] = useState(!!initial.error);
  const submit = async (value: Pending) => {
    if (busy || invalid) return;
    setBusy(true);
    setError(null);
    try {
      sessionStorage.setItem(key, JSON.stringify(value));
      setPending(value);
      const receipt = await canvasApi.resourceCommand(
        canvasId,
        value.operation,
        value.command,
      );
      sessionStorage.removeItem(key);
      setPending(null);
      cache.invalidate();
      onCommitted(receipt);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  return {
    pending,
    error,
    busy,
    invalid,
    run: (operation: Operation, command: Record<string, unknown>) => {
      if (pending) {
        setError("请先确认上次操作结果");
        return;
      }
      void submit({
        operation,
        command: { ...command, requestId: crypto.randomUUID() },
      });
    },
    retry: () => {
      if (pending) void submit(pending);
    },
    discard: () => {
      if (busy) return;
      try {
        sessionStorage.removeItem(key);
        setPending(null);
        setInvalid(false);
        setError(null);
      } catch {
        setError("浏览器无法清理操作记录");
      }
    },
  };
}
