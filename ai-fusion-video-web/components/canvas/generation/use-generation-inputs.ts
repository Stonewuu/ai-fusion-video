"use client";

import { useEffect, useState } from "react";
import { http } from "@/lib/api/client";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasGenerationInputs } from "@/lib/canvas/generation";
import { canvasNodeDependencies } from "@/lib/canvas/node-execution";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import { useCanvas } from "../canvas-provider";
import { useCanvasRuns } from "../runs/canvas-runs-provider";

export function useGenerationInputs(nodeId: string) {
  const canvasId = useCanvas((state) => state.canvasId);
  const revision = useCanvas((state) => state.snapshot!.document.revision);
  const contentRevision = useCanvas((state) => state.contentRevision);
  const graph = useCanvas((state) => state.snapshot!.graph);
  const catalog = useCanvas((state) => state.catalog!);
  const resourceRevision = useResourceChangeStore((state) => state.cursor);
  const { activities } = useCanvasRuns();
  const dependencies = new Set(canvasNodeDependencies(graph, nodeId));
  const generatedSources = new Set(
    graph.nodes
      .filter(
        (node) =>
          node.id !== nodeId &&
          dependencies.has(node.id) &&
          !["textCombine", "collection"].includes(node.kind) &&
          !["constant", "resource"].includes(
            catalog.nodes.find((item) => item.kind === node.kind)!.executor,
          ),
      )
      .map((node) => node.id),
  );
  const runsKey = activities
    .filter(({ node }) => generatedSources.has(node.nodeId))
    .map(({ node }) => `${node.id}:${node.revision}:${node.derivedState}`)
    .join("|");
  const [attempt, setAttempt] = useState(0);
  const key = `${canvasId}:${nodeId}:${revision}:${contentRevision}:${resourceRevision}:${runsKey}:${attempt}`;
  const [loaded, setLoaded] = useState<{
    key: string;
    value: CanvasGenerationInputs | null;
    error: string | null;
  } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    void http
      .get<never, CanvasGenerationInputs>(
        `/api/canvases/${encodeURIComponent(canvasId)}/nodes/${encodeURIComponent(nodeId)}/generation-inputs`,
        { params: { expectedRevision: revision }, signal: controller.signal },
      )
      .then((value) => {
        if (!controller.signal.aborted) {
          setLoaded({ key, value, error: null });
        }
      })
      .catch((failure) => {
        if (!controller.signal.aborted) {
          setLoaded({ key, value: null, error: getApiErrorMessage(failure) });
        }
      });
    return () => controller.abort();
  }, [canvasId, nodeId, revision, key]);
  const current = loaded?.key === key ? loaded : null;
  return {
    value: current?.value ?? null,
    error: current?.error ?? null,
    loading: !current,
    refresh: () => setAttempt((value) => value + 1),
  };
}
