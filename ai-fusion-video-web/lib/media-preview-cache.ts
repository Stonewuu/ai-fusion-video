import { mediaPreviewApi, type MediaPreview } from "@/lib/api/media-preview";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";

export interface PreviewSnapshot {
  view: MediaPreview | null;
  loading: boolean;
  error: string | null;
}
export const EMPTY_PREVIEW: PreviewSnapshot = {
  view: null,
  loading: false,
  error: null,
};
const PENDING: PreviewSnapshot = { view: null, loading: true, error: null };
interface Entry {
  id: string;
  snapshot: PreviewSnapshot;
  listeners: Set<() => void>;
  controller: AbortController;
  timer?: ReturnType<typeof setTimeout>;
  busy: boolean;
  failures: number;
}
class PreviewProtocolError extends Error {}
const entries = new Map<string, Entry>();
const pendingStates = new Set(["QUEUED", "WAITING", "RUNNING"]);

export function previewSnapshot(key: string) {
  return key ? (entries.get(key)?.snapshot ?? PENDING) : EMPTY_PREVIEW;
}
function update(entry: Entry, snapshot: PreviewSnapshot) {
  if (
    entry.controller.signal.aborted ||
    JSON.stringify(snapshot) === JSON.stringify(entry.snapshot)
  )
    return;
  entry.snapshot = snapshot;
  entry.listeners.forEach((notify) => notify());
}
async function request(
  entry: Entry,
  admit: boolean,
  retryId: string | null = null,
) {
  if (entry.busy || entry.controller.signal.aborted) return;
  entry.busy = true;
  clearTimeout(entry.timer);
  try {
    let view = admit
      ? await mediaPreviewApi.ensure(entry.id, retryId, entry.controller.signal)
      : await mediaPreviewApi.get(entry.id, entry.controller.signal);
    if (entry.controller.signal.aborted) return;
    if (view.status === "ABSENT")
      view = await mediaPreviewApi.ensure(
        entry.id,
        null,
        entry.controller.signal,
      );
    if (entry.controller.signal.aborted) return;
    if (
      view.sourceId !== entry.id ||
      view.profile !== "canvas_640_v1" ||
      !view.jobId ||
      !["SUCCEEDED", "FAILED", "CANCELLED", ...pendingStates].includes(
        view.status,
      ) ||
      (view.status === "SUCCEEDED" &&
        (!view.previewId ||
          view.previewId === entry.id ||
          !view.width ||
          !view.height ||
          view.width > 640 ||
          view.height > 480))
    )
      throw new PreviewProtocolError("预览响应与请求的素材不一致");
    const pending = pendingStates.has(view.status);
    entry.failures = 0;
    update(entry, {
      view,
      loading: pending,
      error: view.canRetry ? (view.error ?? "预览生成未完成") : null,
    });
    if (pending)
      entry.timer = setTimeout(() => void request(entry, false), 2000);
  } catch (failure) {
    if (!entry.controller.signal.aborted) {
      update(entry, {
        view: null,
        loading: false,
        error: getApiErrorMessage(failure),
      });
      const status = getApiErrorStatus(failure);
      if (
        !(failure instanceof PreviewProtocolError) &&
        (status === null || status >= 500 || status === 429)
      )
        entry.timer = setTimeout(
          () => void request(entry, false),
          Math.min(30000, 2000 * 2 ** entry.failures++),
        );
    }
  } finally {
    entry.busy = false;
  }
}
export function retryPreview(key: string) {
  const entry = entries.get(key);
  if (!entry || entry.busy) return;
  const previous = entry.snapshot.view;
  update(entry, { view: null, loading: true, error: null });
  void request(entry, true, previous?.canRetry ? previous.jobId : null);
}
export function subscribePreview(
  key: string,
  id: string | null,
  notify: () => void,
) {
  if (!key || !id) return () => {};
  let entry = entries.get(key);
  if (!entry) {
    entry = {
      id,
      snapshot: PENDING,
      listeners: new Set(),
      controller: new AbortController(),
      busy: false,
      failures: 0,
    };
    entries.set(key, entry);
  }
  const current = entry;
  current.listeners.add(notify);
  if (current.listeners.size === 1) void request(current, true);
  return () => {
    if (!current.listeners.delete(notify)) return;
    if (current.listeners.size) return;
    current.controller.abort();
    clearTimeout(current.timer);
    entries.delete(key);
  };
}
