"use client";

import { useCallback, useSyncExternalStore } from "react";
import { mediaId } from "@/lib/api/media";
import { resolveMediaUrl } from "@/lib/api/client";
import { EMPTY_MEDIA_ACCESS, mediaAccessSnapshot, subscribeMediaAccess } from "@/lib/media-access-cache";
import { useAuthStore } from "@/lib/store/auth-store";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";

export function useMediaUrl(source: string | null | undefined, download = false) {
  const userId = useAuthStore((state) => state.user?.id);
  const scope = useResourceChangeStore((state) => state.scope);
  const id = mediaId(source);
  const key = id && userId ? `${userId}:${scope}:${id}:${download}` : "";
  const subscribe = useCallback((notify: () => void) => subscribeMediaAccess(key, id, download, notify), [key, id, download]);
  const snapshot = useSyncExternalStore(subscribe, useCallback(() => mediaAccessSnapshot(key), [key]), () => EMPTY_MEDIA_ACCESS);
  if (!id) return { url: resolveMediaUrl(source), error: null, loading: false };
  return snapshot;
}
