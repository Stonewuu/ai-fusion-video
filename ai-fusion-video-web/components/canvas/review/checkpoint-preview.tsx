"use client";
import { canvasAriaLabels } from "@/lib/canvas/accessibility-labels";

import { useEffect, useMemo, useState } from "react";
import { useTheme } from "next-themes";
import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  ReactFlowProvider,
  type NodeProps,
  type Node,
} from "@xyflow/react";
import {
  Dialog,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { Button } from "@/components/ui/button";
import { canvasApi } from "@/lib/api/canvas";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import {
  nodeMedia,
  nodeSummary,
  type CanvasMediaPreview,
} from "@/lib/canvas/presentation";
import type {
  CanvasCheckpointDetail,
  CanvasNode,
  CanvasResource,
} from "@/lib/canvas/types";
import { useCanvas } from "../canvas-provider";
import { CanvasResourceRead } from "../forms/resource-editor";
import { CanvasMediaPreviewDialog } from "../preview/media-preview";
import { CanvasCheckpointActions } from "./checkpoint-actions";

type SnapshotNode = Node<
  {
    node: CanvasNode;
    resource: CanvasResource | null;
    onOpen: (id: string) => void;
  },
  "snapshot"
>;
function SnapshotNodeCard({ data }: NodeProps<SnapshotNode>) {
  const catalog = useCanvas((state) => state.catalog!);
  const capability = catalog.nodes.find(
    (item) => item.kind === data.node.kind,
  )!;
  return (
    <div className="rounded-xl border border-border/40 bg-card p-3 text-card-foreground">
      <Button
        variant="ghost"
        size="sm"
        className="w-full justify-start"
        onClick={() => data.onOpen(data.node.id)}
      >
        {data.node.title}
      </Button>
      <Handle
        id="annotation-in"
        type="target"
        position={Position.Left}
        isConnectable={false}
        className="invisible"
      />
      <Handle
        id="annotation-out"
        type="source"
        position={Position.Right}
        isConnectable={false}
        className="invisible"
      />
      <p className="text-xs text-muted-foreground">
        {capability.label}
        {data.resource ? ` · 版本 ${data.resource.header.version}` : ""}
      </p>
      <p className="mt-2 line-clamp-4 whitespace-pre-wrap text-xs">
        {nodeSummary(data.node, data.resource)}
      </p>
      {capability.inputs.map((port, index) => (
        <Handle
          key={port.id}
          id={port.id}
          type="target"
          position={Position.Left}
          isConnectable={false}
          style={{ top: 24 + index * 12 }}
        />
      ))}
      {capability.outputs.map((port, index) => (
        <Handle
          key={port.id}
          id={port.id}
          type="source"
          position={Position.Right}
          isConnectable={false}
          style={{ top: 24 + index * 12 }}
        />
      ))}
    </div>
  );
}
const nodeTypes = { snapshot: SnapshotNodeCard };
export function CanvasCheckpointPreview({
  id,
  onClose,
}: {
  id: string;
  onClose: () => void;
}) {
  const canvasId = useCanvas((state) => state.canvasId);
  const { resolvedTheme } = useTheme();
  const scope = useResourceChangeStore((state) => state.scope);
  const revision = useResourceChangeStore((state) => state.cursor);
  const key = `${id}:${scope}:${revision}`;
  const [result, setResult] = useState<{
    key: string;
    data?: CanvasCheckpointDetail;
    error?: string;
  } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [preview, setPreview] = useState<CanvasMediaPreview | null>(null);
  useEffect(() => {
    let active = true;
    void canvasApi
      .checkpoint(canvasId, id)
      .then((data) => {
        if (active) setResult({ key, data });
      })
      .catch((error) => {
        if (active) setResult({ key, error: getApiErrorMessage(error) });
      });
    return () => {
      active = false;
    };
  }, [canvasId, id, key]);
  const data = result?.key === key ? result.data : undefined;
  const nodes = useMemo<SnapshotNode[]>(() => {
    if (!data) return [];
    const graph = data.snapshot.graph;
    const depth = (node: CanvasNode): number => {
      let count = 0;
      let parent = node.layout.parentId;
      while (parent) {
        count++;
        parent =
          graph.nodes.find((item) => item.id === parent)?.layout.parentId ??
          null;
      }
      return count;
    };
    return [...graph.nodes]
      .sort((a, b) => depth(a) - depth(b))
      .map((node) => ({
        id: node.id,
        type: "snapshot",
        position: { x: node.layout.x, y: node.layout.y },
        parentId: node.layout.parentId ?? undefined,
        style: {
          width: node.layout.width ?? 300,
          height:
            node.kind === "group" ? (node.layout.height ?? 400) : undefined,
        },
        data: {
          node,
          onOpen: setSelected,
          resource:
            data.snapshot.resources.find(
              (resource) =>
                resource.header.resourceType === node.resource?.type &&
                resource.header.resourceId === node.resource?.id,
            ) ?? null,
        },
      }));
  }, [data]);
  const selectedData = nodes.find((node) => node.id === selected)?.data;
  const selectedResource = selectedData?.resource;
  const selectedMedia = selectedData
    ? nodeMedia(selectedData.node, selectedData.resource)
    : null;
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="flex h-[80vh] max-h-[calc(100vh-2rem)] sm:max-w-6xl flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>{data?.checkpoint.title ?? "命名版本"}</DialogTitle>
          <DialogDescription>
            正在查看保存时的版本。点击内容节点可阅读当时的内容。
          </DialogDescription>
        </DialogHeader>
        {data && (
          <CanvasCheckpointActions
            id={id}
            title={data.checkpoint.title}
            onApplied={onClose}
          />
        )}
        {result?.key === key && result.error ? (
          <p role="alert" className="text-sm text-destructive">
            {result.error}
          </p>
        ) : !data ? (
          <p role="status">正在读取命名版本…</p>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col gap-4 md:flex-row">
            <div className="min-h-48 min-w-0 flex-1 rounded-xl border border-border/30">
              <ReactFlowProvider>
                <ReactFlow<SnapshotNode>
                  defaultNodes={nodes}
                  className="fusion-canvas"
                  colorMode={resolvedTheme === "dark" ? "dark" : "light"}
                  ariaLabelConfig={{
                    ...canvasAriaLabels,
                    "node.a11yDescription.default": "按回车或空格选择节点，查看保存时的内容。",
                    "edge.a11yDescription.default": "这是保存时的连线，只能查看。",
                    "controls.zoomIn.ariaLabel": "放大快照",
                    "controls.zoomOut.ariaLabel": "缩小快照",
                    "controls.fitView.ariaLabel": "查看完整快照",
                  }}
                  nodeTypes={nodeTypes}
                  defaultEdges={data.snapshot.graph.edges.map((edge) => ({
                    id: edge.id,
                    source: edge.sourceNodeId,
                    target: edge.targetNodeId,
                    sourceHandle:
                      edge.kind === "execution"
                        ? edge.sourcePort
                        : "annotation-out",
                    targetHandle:
                      edge.kind === "execution"
                        ? edge.targetPort
                        : "annotation-in",
                  }))}
                  nodesDraggable={false}
                  nodesConnectable={false}
                  edgesReconnectable={false}
                  deleteKeyCode={null}
                  onNodeClick={(_event, node) => setSelected(node.id)}
                  fitView
                  fitViewOptions={{ padding: 0.18, maxZoom: 1 }}
                >
                  <Background />
                  <div role="group" aria-label="版本预览缩放控制">
                    <Controls
                      aria-label=""
                      showInteractive={false}
                      fitViewOptions={{ padding: 0.18, maxZoom: 1 }}
                    />
                  </div>
                </ReactFlow>
              </ReactFlowProvider>
            </div>
            {selectedData && (
              <div className="max-h-60 min-h-0 overflow-y-auto md:max-h-none md:w-80">
                {selectedResource ? (
                  <CanvasResourceRead
                    resource={selectedResource}
                    library={[]}
                    onPreview={setPreview}
                  />
                ) : (
                  <div className="space-y-3 p-3">
                    <h3 className="text-sm font-semibold">
                      {selectedData.node.title}
                    </h3>
                    <p className="whitespace-pre-wrap break-words text-sm">
                      {nodeSummary(selectedData.node, null)}
                    </p>
                  </div>
                )}
                {selectedMedia && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPreview(selectedMedia)}
                  >
                    预览媒体
                  </Button>
                )}
              </div>
            )}
          </div>
        )}
        {preview && (
          <CanvasMediaPreviewDialog
            media={preview}
            onClose={() => setPreview(null)}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}
