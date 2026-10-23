import type { CanvasRun } from "@/lib/api/canvas-runs";
import { runFinished } from "./run-state";

/** Only finished runs can be retried; unresolved submissions remain attached to their original request. */
export function canvasRecoveryPlan(run: CanvasRun) {
  if (!run.canControl || !runFinished(run.status) || run.status === "SUCCEEDED")
    return null;
  const nodes = run.nodes.filter(
    (node) =>
      node.parentNodeRunId === null && node.derivedState !== "NODE_DELETED",
  );
  const retry = nodes.filter(
    (node) =>
      ["FAILED", "BLOCKED", "CANCELLED", "PARTIAL_FAILED"].includes(
        node.status,
      ) && !node.contentUnavailable,
  );
  if (!retry.length) return null;
  const reuse = Object.fromEntries(
    nodes
      .filter(
        (node) =>
          node.status === "SUCCEEDED" &&
          node.derivedState === "CURRENT" &&
          !node.contentUnavailable,
      )
      .map((node) => [node.nodeId, node.id]),
  );
  return {
    mode: "DEPENDENCIES" as const,
    nodeIds: retry.map((node) => node.nodeId),
    reuse,
  };
}
