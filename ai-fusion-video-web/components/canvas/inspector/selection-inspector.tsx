"use client";

import {
  GitBranch,
  Layers,
  LocateFixed,
  MousePointer2,
  X,
} from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import { CanvasNodeTypeIcon } from "../shared/canvas-node-icon";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";

export function CanvasSelectionInspector({
  onInspect,
  onInspectEdge,
}: {
  onInspect: (id: string) => void;
  onInspectEdge: (id: string) => void;
}) {
  const store = useCanvasStoreApi();
  const graph = useCanvas((state) => state.snapshot!.graph);
  const catalog = useCanvas((state) => state.catalog!);
  const selection = useCanvas((state) => state.selection);
  const edgeSelection = useCanvas((state) => state.edgeSelection);
  const nodes = graph.nodes.filter((node) => selection.includes(node.id));
  const edges = graph.edges.filter((edge) => edgeSelection.includes(edge.id));
  const total = nodes.length + edges.length;

  return (
    <section className="flex flex-col gap-3 p-3.5" aria-label="当前选择">
      {total === 0 ? (
        <div className="flex flex-col items-center gap-2 rounded-[12px] border border-dashed border-foreground/[0.1] bg-foreground/[0.02] px-4 py-8 text-center dark:border-white/[0.1] dark:bg-white/[0.02]">
          <span className="grid size-9 place-items-center rounded-[10px] bg-foreground/[0.05] text-muted-foreground dark:bg-white/[0.07]">
            <MousePointer2 className="size-4" aria-hidden />
          </span>
          <h2 className="text-[13px] font-medium">选区与连线</h2>
          <p className="text-xs leading-relaxed text-muted-foreground">
            点击单个节点可在节点下方直接编辑；点击连线或框选多个节点时可在这里查看与定位。
          </p>
        </div>
      ) : (
        <>
          <div className="flex flex-col gap-2.5 rounded-[12px] border border-foreground/[0.07] bg-foreground/[0.02] p-3 dark:border-white/[0.08] dark:bg-white/[0.025]">
            <div className="flex items-center gap-2">
              <span className="grid size-6 shrink-0 place-items-center rounded-[7px] bg-primary/12 text-primary">
                <Layers className="size-3.5" aria-hidden />
              </span>
              <h2 className="min-w-0 flex-1 text-[13px] font-medium">
                已选 {nodes.length} 个节点
                {edges.length > 0 && ` · ${edges.length} 条连线`}
              </h2>
            </div>
            <p className="text-xs leading-relaxed text-muted-foreground">
              点击下方条目可聚焦并展开对应编辑面板。
            </p>
            <div className="flex flex-wrap items-center gap-1.5">
              {nodes.length > 0 && (
                <CanvasButton
                  variant="secondary"
                  size="sm"
                  onClick={() => store.getState().focus(selection)}
                >
                  <LocateFixed />
                  定位所选
                </CanvasButton>
              )}
              <CanvasButton
                variant="ghost"
                size="sm"
                onClick={() => store.getState().select([])}
              >
                <X />
                取消选择
              </CanvasButton>
            </div>
          </div>
          <ul className="flex flex-col gap-0.5">
            {nodes.map((node) => {
              const capability = catalog.nodes.find(
                (item) => item.kind === node.kind,
              );
              return (
                <li key={node.id}>
                  <CanvasButton
                    variant="ghost"
                    className="w-full justify-start"
                    onClick={() => onInspect(node.id)}
                  >
                    {capability && (
                      <CanvasNodeTypeIcon
                        capability={capability}
                        aria-hidden
                        style={{ color: canvasNodeColor(node.kind) }}
                      />
                    )}
                    <span className="min-w-0 flex-1 truncate text-left">
                      {node.title}
                    </span>
                    <span className="shrink-0 text-[11px] text-muted-foreground">
                      {capability?.label}
                    </span>
                  </CanvasButton>
                </li>
              );
            })}
            {edges.map((edge) => (
              <li key={edge.id}>
                <CanvasButton
                  variant="ghost"
                  className="w-full justify-start"
                  onClick={() => onInspectEdge(edge.id)}
                >
                  <GitBranch className="text-muted-foreground" aria-hidden />
                  <span className="min-w-0 flex-1 truncate text-left">
                    {
                      graph.nodes.find((node) => node.id === edge.sourceNodeId)
                        ?.title
                    }
                    {" → "}
                    {
                      graph.nodes.find((node) => node.id === edge.targetNodeId)
                        ?.title
                    }
                  </span>
                </CanvasButton>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
