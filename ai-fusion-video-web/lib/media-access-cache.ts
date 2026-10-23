import { accessMedia } from "@/lib/api/media";
import { resolveMediaUrl } from "@/lib/api/client";
import { getApiErrorMessage } from "@/lib/api/api-error";

export interface MediaAccessSnapshot { url: string | null; error: string | null; loading: boolean }
export const EMPTY_MEDIA_ACCESS: MediaAccessSnapshot = { url: null, error: null, loading: false };
const PENDING: MediaAccessSnapshot = { url: null, error: null, loading: true };
interface Entry {
  snapshot: MediaAccessSnapshot;
  listeners: Set<() => void>;
  controller: AbortController;
  timer?: ReturnType<typeof setTimeout>;
}
const active = new Map<string, Entry>();

export function mediaAccessSnapshot(key: string) {
  return key ? active.get(key)?.snapshot ?? PENDING : EMPTY_MEDIA_ACCESS;
}

export function subscribeMediaAccess(key: string, id: string | null, download: boolean, notify: () => void) {
  if (!key || !id) return () => {};
  let entry = active.get(key);
  if (!entry) {
    entry = { snapshot: PENDING, listeners: new Set(), controller: new AbortController() };
    active.set(key, entry);
  }
  entry.listeners.add(notify);
  const current = entry;
  const issue = async () => {
    try {
      const access = await accessMedia(id, download, current.controller.signal);
      if (current.controller.signal.aborted) return;
      const remaining = Date.parse(access.expiresAt) - Date.now();
      const url = resolveMediaUrl(access.url);
      if (!url || !Number.isFinite(remaining) || remaining <= 1000) throw new Error("媒体访问凭据已失效");
      current.snapshot = { url, loading: false, error: null };
      current.timer = setTimeout(issue, Math.max(1000, remaining - 30000));
    } catch (error) {
      if (current.controller.signal.aborted) return;
      current.snapshot = { url: null, loading: false, error: getApiErrorMessage(error) };
      current.timer = setTimeout(issue, 30000);
    }
    current.listeners.forEach((listener) => listener());
  };
  if (entry.listeners.size === 1) void issue();
  return () => {
    current.listeners.delete(notify);
    if (current.listeners.size !== 0) return;
    current.controller.abort();
    clearTimeout(current.timer);
    active.delete(key);
  };
}
