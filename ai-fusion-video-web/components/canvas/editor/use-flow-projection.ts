"use client";
import { useMemo } from "react";
import type { Edge } from "@xyflow/react";
import type { CanvasCatalog, CanvasSnapshot } from "@/lib/canvas/types";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import type { CanvasFlowNode } from "./canvas-node";
import type { Transient, Measurements } from "./use-flow-motion";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import { canvasNodeConfigurationIssue } from "@/lib/canvas/node-readiness";
import { canvasConnectionIssue } from "@/lib/canvas/connection-readiness";
import {
  canvasGraphHierarchy,
  canvasNodeGeometry,
} from "@/lib/canvas/flow-geometry";
interface Props {
  snapshot: CanvasSnapshot;
  catalog: CanvasCatalog;
  selection: string[];
  edgeSelection: string[];
  transient: Record<string, Transient>;
  measurements: Measurements;
  editable: boolean;
  pointerMode: "select" | "pan";
  onPreview: (media: CanvasMediaPreview, returnFocus?: HTMLElement) => void;
  onRunNode: CanvasFlowNode["data"]["onRunNode"];
  onShowNodeRun: CanvasFlowNode["data"]["onShowNodeRun"];
  addAfter: CanvasFlowNode["data"]["onAddAfter"];
  focusRequest: { ids: string[]; sequence: number } | null;
  agentChangedIds: string[];
}
export function useCanvasFlowProjection({
  snapshot,
  catalog,
  selection,
  edgeSelection,
  transient,
  measurements,
  editable,
  pointerMode,
  onPreview,
  onRunNode,
  onShowNodeRun,
  addAfter,
  focusRequest,
  agentChangedIds,
}: Props) {
  const hierarchy = useMemo(
    () => canvasGraphHierarchy(snapshot.graph.nodes),
    [snapshot.graph.nodes],
  );
  const capabilities = useMemo(
    () => new Map(catalog.nodes.map((node) => [node.kind, node])),
    [catalog.nodes],
  );
  const focusedGroups = useMemo(() => {
    const result = new Set<string>();
    for (const id of focusRequest?.ids ?? []) {
      let parent = hierarchy.byId.get(id)?.layout.parentId;
      while (parent) {
        result.add(parent);
        parent = hierarchy.byId.get(parent)?.layout.parentId;
      }
    }
    return result;
  }, [focusRequest, hierarchy]);
  const configurationIssues = useMemo(() => new Map(snapshot.graph.nodes.map((node) => [
    node.id, canvasNodeConfigurationIssue(node, capabilities.get(node.kind)!, catalog, snapshot.graph),
  ])), [capabilities, catalog, snapshot.graph]);
  const connectionIssues = useMemo(() => new Map(snapshot.graph.edges.map((edge) => [
    edge.id, canvasConnectionIssue(edge, snapshot.graph, catalog),
  ])), [catalog, snapshot.graph]);
  const invalidNodes = useMemo(() => new Map(snapshot.graph.edges.filter((edge) => connectionIssues.get(edge.id))
    .map((edge) => [edge.targetNodeId, connectionIssues.get(edge.id)!])), [connectionIssues, snapshot.graph.edges]);
  const hidden = useMemo(() => {
    const result = new Set<string>();
    for (const node of hierarchy.ordered) {
      const parent = node.layout.parentId;
      if (
        parent &&
        (result.has(parent) ||
          (hierarchy.byId.get(parent)!.layout.collapsed &&
            !focusedGroups.has(parent)))
      ) {
        result.add(node.id);
      }
    }
    return result;
  }, [hierarchy, focusedGroups]);
  const nodes = useMemo<CanvasFlowNode[]>(() => {
    const selected = new Set(selection);
    const changed = new Set(agentChangedIds);
    return hierarchy.ordered.map((node) => ({
      id: node.id,
      type: "canvas",
      ...canvasNodeGeometry(
        node,
        capabilities.get(node.kind)!,
        node.layout.collapsed && !focusedGroups.has(node.id),
        { ...measurements[node.id], ...transient[node.id] },
      ),
      className: changed.has(node.id) ? "canvas-agent-changed" : undefined,
      measured: measurements[node.id],
      position: transient[node.id]?.position ?? {
        x: node.layout.x,
        y: node.layout.y,
      },
      parentId: node.layout.parentId ?? undefined,
      hidden: hidden.has(node.id),
      selected: selected.has(node.id),
      draggable: editable && pointerMode === "select" && !node.layout.locked,
      dragHandle: ".canvas-node-drag",
      ariaLabel: `${node.title}，${capabilities.get(node.kind)!.label}`,
      style: {
        width: transient[node.id]?.width ?? node.layout.width ?? 300,
        height:
          node.kind === "group"
            ? node.layout.collapsed && !focusedGroups.has(node.id)
              ? 64
              : (transient[node.id]?.height ?? node.layout.height ?? 400)
            : undefined,
      },
      zIndex: node.kind === "group" ? -1 : 0,
      data: {
        node,
        capability: capabilities.get(node.kind)!,
        canEdit: !!editable,
        childCount: hierarchy.childCounts.get(node.id) ?? 0,
        configurationIssue: invalidNodes.get(node.id) ?? configurationIssues.get(node.id) ?? null,
        invalidInput: invalidNodes.has(node.id),
        onPreview,
        onRunNode,
        onShowNodeRun,
        onAddAfter: addAfter,
      },
    }));
  }, [
    hierarchy,
    capabilities,
    configurationIssues,
    invalidNodes,
    selection,
    transient,
    measurements,
    hidden,
    editable,
    pointerMode,
    onPreview,
    onRunNode,
    onShowNodeRun,
    addAfter,
    focusedGroups,
    agentChangedIds,
  ]);
  const edges = useMemo<Edge[]>(
    () =>
      snapshot.graph.edges.map((edge) => ({
        id: edge.id,
        source: edge.sourceNodeId,
        target: edge.targetNodeId,
        sourceHandle:
          edge.kind === "execution" ? edge.sourcePort : "annotation-out",
        targetHandle:
          edge.kind === "execution" ? edge.targetPort : "annotation-in",
        hidden: hidden.has(edge.sourceNodeId) || hidden.has(edge.targetNodeId),
        selected: edgeSelection.includes(edge.id),
        type: "canvas",
        data: { warning: connectionIssues.get(edge.id) },
        ariaLabel: connectionIssues.get(edge.id) ? `连线需要调整：${connectionIssues.get(edge.id)}` : undefined,
        reconnectable: editable && edge.kind === "execution",
        animated: false,
        label:
          connectionIssues.get(edge.id) ? "检查输入" : edge.kind === "execution" &&
          edge.inputIndex !== null &&
          edge.inputIndex > 0
            ? String(edge.inputIndex + 1)
            : undefined,
        style: {
          stroke: connectionIssues.get(edge.id) ? "var(--destructive)" : edge.kind === "annotation"
              ? "var(--muted-foreground)"
              : canvasNodeColor(hierarchy.byId.get(edge.sourceNodeId)!.kind),
          strokeDasharray: connectionIssues.get(edge.id) ? "6 4" : edge.kind === "execution" ? undefined : "5 4",
          strokeWidth: connectionIssues.get(edge.id) ? 2 : undefined,
        },
        labelStyle: connectionIssues.get(edge.id) ? { fill: "var(--destructive)", fontSize: 11 } : undefined,
      })),
    [snapshot.graph.edges, hierarchy, edgeSelection, hidden, editable, connectionIssues],
  );
  return { nodes, edges, hidden };
}
