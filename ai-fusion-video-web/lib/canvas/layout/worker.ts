import { arrangeWithBrowserEngine } from "./elk-engine";
import type { LayoutRequest, LayoutResult } from "./types";

export interface LayoutWorkerMessage {
  request: LayoutRequest;
}

export type LayoutWorkerResponse = { result: Pick<LayoutResult, "positions"> } | { error: string };

self.onmessage = async (event: MessageEvent<LayoutWorkerMessage>) => {
  try {
    const { positions } = await arrangeWithBrowserEngine(event.data.request);
    self.postMessage({ result: { positions } } satisfies LayoutWorkerResponse);
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.message : String(error) } satisfies LayoutWorkerResponse);
  }
};
