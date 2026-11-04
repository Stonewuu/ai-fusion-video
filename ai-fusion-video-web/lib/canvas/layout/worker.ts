import { routeCanvasEdges } from "./routing";
import type { LayoutRequest, LayoutResult } from "./types";

export interface LayoutWorkerMessage {
  mode: "arrange" | "route";
  request: LayoutRequest;
}

export type LayoutWorkerResponse = { result: LayoutResult } | { error: string };

self.onmessage = async (event: MessageEvent<LayoutWorkerMessage>) => {
  try {
    const { mode, request } = event.data;
    const result = mode === "arrange"
      ? await (await import("./elk-engine")).arrangeWithBrowserEngine(request)
      : { positions: {}, routes: routeCanvasEdges(request) };
    self.postMessage({ result } satisfies LayoutWorkerResponse);
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.message : String(error) } satisfies LayoutWorkerResponse);
  }
};
