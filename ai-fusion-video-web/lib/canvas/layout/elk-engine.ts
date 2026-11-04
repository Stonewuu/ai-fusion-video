import ELK from "elkjs/lib/elk-api.js";
import { autoArrangeCanvas } from "../auto-layout";
import type { LayoutRequest } from "./types";

export async function arrangeWithBrowserEngine(request: LayoutRequest) {
  // ELK 独立处理自己的消息，避免合并包接管外层布局 Worker 的通信。
  const engine = new ELK({
    algorithms: ["layered"],
    workerFactory: () => new Worker(new URL("./elk-worker.ts", import.meta.url), { type: "module" }),
  });
  try {
    return await autoArrangeCanvas(request, engine);
  } finally {
    engine.terminateWorker();
  }
}
