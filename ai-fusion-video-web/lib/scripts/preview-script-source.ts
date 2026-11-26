import type { ScriptSourcePreview } from "@/lib/api/script";
import type { SourceSplitRequest, SourceSplitResponse } from "./source-split.worker";

export function previewScriptSource(request: SourceSplitRequest, signal: AbortSignal) {
  return new Promise<ScriptSourcePreview>((resolve, reject) => {
    signal.throwIfAborted();
    let worker: Worker;
    const unavailable = "无法在浏览器中识别分集，请刷新页面后重试";
    try {
      worker = new Worker(new URL("./source-split.worker.ts", import.meta.url), { type: "module" });
    } catch {
      reject(new Error(unavailable));
      return;
    }
    const finish = () => {
      signal.removeEventListener("abort", abort);
      worker.terminate();
    };
    const abort = () => {
      finish();
      reject(signal.reason);
    };
    signal.addEventListener("abort", abort, { once: true });
    worker.onmessage = ({ data }: MessageEvent<SourceSplitResponse>) => {
      finish();
      if ("error" in data) {
        reject(new Error(data.error));
      } else {
        resolve(data.preview);
      }
    };
    const fail = () => {
      finish();
      reject(new Error(unavailable));
    };
    worker.onerror = fail;
    worker.onmessageerror = fail;
    try {
      worker.postMessage(request);
    } catch {
      fail();
    }
  });
}
