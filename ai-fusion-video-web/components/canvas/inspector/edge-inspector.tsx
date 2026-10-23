"use client";
import { useState } from "react";
import { ArrowDown, ArrowUp, Locate, Unlink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { canvasPortLabels } from "@/lib/canvas/labels";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasFixedOutput } from "./fixed-output";
import { canvasConnectionIssue } from "@/lib/canvas/connection-readiness";
import { CanvasGuidance } from "../shared/canvas-guidance";

export function CanvasEdgeInspector({ edgeId }: { edgeId: string }) {
  const store = useCanvasStoreApi();
  const snapshot = useCanvas((state) => state.snapshot!);
  const pending = useCanvas((state) => !!state.pending);
  const catalog = useCanvas((state) => state.catalog!);
  const [error, setError] = useState<string | null>(null);
  const edge = snapshot.graph.edges.find((item) => item.id === edgeId);
  if (!edge) return <p className="p-4 text-sm">此连线已移除。</p>;
  const source = snapshot.graph.nodes.find(
    (item) => item.id === edge.sourceNodeId,
  )!;
  const target = snapshot.graph.nodes.find(
    (item) => item.id === edge.targetNodeId,
  )!;
  const inputs = snapshot.graph.edges
    .filter(
      (item) =>
        item.kind === "execution" &&
        item.targetNodeId === edge.targetNodeId &&
        item.targetPort === edge.targetPort,
    )
    .sort((a, b) => (a.inputIndex ?? 0) - (b.inputIndex ?? 0));
  const editable =
    snapshot.permissions.canEdit && !snapshot.document.archived && !pending;
  const configurationIssue = canvasConnectionIssue(edge, snapshot.graph, catalog);
  const move = async (index: number, direction: number) => {
    const a = inputs[index],
      b = inputs[index + direction];
    try {
      await store.getState().edit("调整输入顺序", [
        {
          type: "canvas.edge.replace",
          edge: { ...a, inputIndex: b.inputIndex },
          expectedVersion: a.version,
        },
        {
          type: "canvas.edge.replace",
          edge: { ...b, inputIndex: a.inputIndex },
          expectedVersion: b.version,
        },
      ]);
      setError(null);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    }
  };
  return (
    <div className="space-y-5 p-4">
      {configurationIssue && <CanvasGuidance title="输入需要调整" tone="error">
        {configurationIssue}。连线已保留，调整后即可运行。
        <Button variant="outline" size="sm" onClick={() => store.getState().focus([target.id])}>配置目标节点</Button>
      </CanvasGuidance>}
      <div className="space-y-2">
        <h2 className="text-sm font-semibold">
          {edge.kind === "execution" ? "输入连接" : "说明连线"}
        </h2>
        <p className="text-xs text-muted-foreground">
          {edge.kind === "execution"
            ? edge.output
              ? "此连线使用固定结果；调整输入顺序不会改变固定内容。"
              : "运行时按以下顺序接收前一步的内容；移动节点不会改变顺序。"
            : "用于解释节点之间的联系，不会生成内容或修改项目关联。"}
        </p>
      </div>
      <div className="space-y-2">
        <p className="break-words text-sm">
          {source.title} ·{" "}
          {edge.sourcePort
            ? (canvasPortLabels[edge.sourcePort] ?? edge.sourcePort)
            : "说明"}
        </p>
        <p className="break-words text-sm">
          → {target.title} ·{" "}
          {edge.targetPort
            ? (canvasPortLabels[edge.targetPort] ?? edge.targetPort)
            : "说明"}
        </p>
        <Button
          variant="outline"
          size="sm"
          onClick={() => store.getState().focus([source.id, target.id])}
        >
          <Locate />
          定位两端
        </Button>
      </div>
      {edge.kind === "execution" && (
        <section className="space-y-2">
          <h3 className="text-sm font-medium">输入顺序</h3>
          {inputs.map((item, index) => (
            <div
              key={item.id}
              className="flex items-center gap-2 border-b border-border/20 py-2"
            >
              <span className="min-w-0 flex-1 truncate text-sm">
                {index + 1}.{" "}
                {
                  snapshot.graph.nodes.find(
                    (node) => node.id === item.sourceNodeId,
                  )?.title
                }
                {item.sourcePort && ` · ${canvasPortLabels[item.sourcePort] ?? item.sourcePort}`}
              </span>
              <Button
                variant="ghost"
                size="icon-xs"
                title="上移输入"
                disabled={!editable || index === 0}
                onClick={() => void move(index, -1)}
              >
                <ArrowUp />
              </Button>
              <Button
                variant="ghost"
                size="icon-xs"
                title="下移输入"
                disabled={!editable || index === inputs.length - 1}
                onClick={() => void move(index, 1)}
              >
                <ArrowDown />
              </Button>
            </div>
          ))}
        </section>
      )}
      {edge.output && <CanvasFixedOutput edge={edge} editable={editable} />}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <Button
        variant="destructive"
        size="sm"
        disabled={!editable}
        onClick={() =>
          void store
            .getState()
            .edit("移除连线", [
              {
                type: "canvas.edge.delete",
                edgeId: edge.id,
                expectedVersion: edge.version,
              },
            ])
            .catch((failure) => setError(getApiErrorMessage(failure)))
        }
      >
        <Unlink />
        移除连线
      </Button>
    </div>
  );
}
