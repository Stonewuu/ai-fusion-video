"use client";
import { useCallback, useSyncExternalStore } from "react";
import { mediaId } from "@/lib/api/media";
import {
  EMPTY_PREVIEW,
  previewSnapshot,
  retryPreview,
  subscribePreview,
} from "@/lib/media-preview-cache";
import { useAuthStore } from "@/lib/store/auth-store";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";

export function useMediaPreview(source: string, enabled: boolean) {
  const userId = useAuthStore((s) => s.user?.id);
  const scope = useResourceChangeStore((s) => s.scope);
  const id = mediaId(source);
  const key = enabled && id && userId ? `${userId}:${scope}:${id}` : "";
  const subscribe = useCallback(
    (notify: () => void) => subscribePreview(key, id, notify),
    [key, id],
  );
  const state = useSyncExternalStore(
    subscribe,
    useCallback(() => previewSnapshot(key), [key]),
    () => EMPTY_PREVIEW,
  );
  return { ...state, retry: () => retryPreview(key) };
}
