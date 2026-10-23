"use client";

import { Layers, LocateFixed, Pencil, MousePointer2 } from "lucide-react";
import { Button } from "@/components/ui/button";
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
  return (
    <section className="space-y-4 p-4" aria-label="当前选择">
      {nodes.length + edges.length === 0 ? (
        <>
          <MousePointer2 className="size-5 text-muted-foreground" />
          <h2 className="text-sm font-semibold">选择要查看的内容</h2>
          <p className="text-sm text-muted-foreground">
            点击画布上的节点，在这里编辑内容、调整设置或预览媒体。
          </p>
        </>
      ) : (
        <>
          <div className="space-y-2">
            <Layers className="size-5 text-muted-foreground" />
            <h2 className="text-sm font-semibold">
              已选 {nodes.length} 个节点
              {edges.length > 0 && `、${edges.length} 条连线`}
            </h2>
            <p className="text-xs text-muted-foreground">
              在画布工具栏批量整理、分组或移除，也可以选择下方一项单独编辑。
            </p>
          </div>
          {nodes.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => store.getState().focus(selection)}
            >
              <LocateFixed />
              定位所选节点
            </Button>
          )}
          <ul className="space-y-2">
            {nodes.map((node) => (
              <li key={node.id}>
                <Button
                  variant="ghost"
                  className="w-full justify-start"
                  onClick={() => onInspect(node.id)}
                >
                  <Pencil />
                  <span className="min-w-0 flex-1 truncate text-left">
                    {node.title}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {
                      catalog.nodes.find((item) => item.kind === node.kind)
                        ?.label
                    }
                  </span>
                </Button>
              </li>
            ))}
            {edges.map((edge) => (
              <li key={edge.id}>
                <Button
                  variant="ghost"
                  className="w-full justify-start"
                  onClick={() => onInspectEdge(edge.id)}
                >
                  <Pencil />
                  <span className="truncate">
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
                </Button>
              </li>
            ))}
          </ul>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => store.getState().select([])}
          >
            取消选择
          </Button>
        </>
      )}
    </section>
  );
}
