import type { LayoutRequest, LayoutResult } from "./types";
import type { LayoutWorkerMessage, LayoutWorkerResponse } from "./worker";

export function runLayoutWorker(request: LayoutRequest, signal: AbortSignal) {
  return new Promise<Pick<LayoutResult, "positions">>((resolve, reject) => {
    signal.throwIfAborted();
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    const finish = () => {
      clearTimeout(timeout);
      signal.removeEventListener("abort", abort);
      worker.terminate();
    };
    const abort = () => {
      finish();
      reject(new DOMException("计算已取消", "AbortError"));
    };
    const timeout = setTimeout(() => {
      finish();
      reject(new Error("整理计算超时，请减少选中的节点后重试"));
    }, 15000);
    signal.addEventListener("abort", abort, { once: true });
    worker.onmessage = (event: MessageEvent<LayoutWorkerResponse>) => {
      finish();
      if ("error" in event.data) {
        reject(new Error(event.data.error));
      } else {
        resolve(event.data.result);
      }
    };
    worker.onerror = () => {
      finish();
      reject(new Error("无法启动画布布局计算，请刷新页面后重试"));
    };
    worker.postMessage({ request } satisfies LayoutWorkerMessage);
  });
}
