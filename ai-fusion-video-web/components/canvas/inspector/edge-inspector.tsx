"use client";

import { useState } from "react";
import { ArrowDown, ArrowRight, ArrowUp, Locate, Unlink } from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
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
  if (!edge) {
    return (
      <p className="p-3.5 text-xs text-muted-foreground">此连线已移除。</p>
    );
  }
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
  const configurationIssue = canvasConnectionIssue(
    edge,
    snapshot.graph,
    catalog,
  );
  const move = async (index: number, direction: number) => {
    const a = inputs[index];
    const b = inputs[index + direction];
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
    <div className="flex flex-col gap-3 p-3.5">
      {configurationIssue && (
        <CanvasGuidance title="输入需要调整" tone="error">
          {configurationIssue}。连线已保留，调整后即可运行。
          <CanvasButton
            variant="secondary"
            size="sm"
            onClick={() => store.getState().focus([target.id])}
          >
            定位目标节点
          </CanvasButton>
        </CanvasGuidance>
      )}
      <div className="flex flex-col gap-2.5 rounded-[12px] border border-foreground/[0.07] bg-foreground/[0.02] p-3 dark:border-white/[0.08] dark:bg-white/[0.025]">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-[13px] font-medium">
            {edge.kind === "execution" ? "输入连接" : "说明连线"}
          </h2>
          <CanvasButton
            variant="secondary"
            size="sm"
            onClick={() => store.getState().focus([source.id, target.id])}
          >
            <Locate />
            定位两端
          </CanvasButton>
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {edge.kind === "execution"
            ? edge.output
              ? "此连线使用固定结果；调整输入顺序不会改变固定内容。"
              : "运行时按顺序接收前一步的内容；移动节点不会改变顺序。"
            : "用于解释节点之间的联系，不会生成内容或修改项目关联。"}
        </p>
        <div className="flex flex-col gap-1.5 rounded-[10px] border border-foreground/[0.06] bg-background/80 px-2.5 py-2 text-xs dark:border-white/[0.08] dark:bg-white/[0.04]">
          <p className="truncate font-medium">
            {source.title}
            <span className="font-normal text-muted-foreground">
              {" · "}
              {edge.sourcePort
                ? (canvasPortLabels[edge.sourcePort] ?? edge.sourcePort)
                : "说明"}
            </span>
          </p>
          <p className="flex items-center gap-1 truncate font-medium">
            <ArrowRight
              className="size-3 shrink-0 text-muted-foreground"
              aria-hidden
            />
            <span className="truncate">{target.title}</span>
            <span className="shrink-0 font-normal text-muted-foreground">
              {" · "}
              {edge.targetPort
                ? (canvasPortLabels[edge.targetPort] ?? edge.targetPort)
                : "说明"}
            </span>
          </p>
        </div>
      </div>
      {edge.kind === "execution" && (
        <section className="flex flex-col gap-1.5">
          <h3 className="px-0.5 text-xs font-medium text-muted-foreground">
            输入顺序
          </h3>
          <div className="flex flex-col gap-1">
            {inputs.map((item, index) => (
              <div
                key={item.id}
                className="flex items-center gap-1.5 rounded-[10px] border border-foreground/[0.07] bg-background/70 px-2.5 py-1.5 dark:border-white/[0.08] dark:bg-white/[0.03]"
              >
                <span className="grid size-5 shrink-0 place-items-center rounded-[6px] bg-foreground/[0.06] text-[11px] font-medium tabular-nums text-muted-foreground dark:bg-white/[0.08]">
                  {index + 1}
                </span>
                <span className="min-w-0 flex-1 truncate text-xs">
                  {
                    snapshot.graph.nodes.find(
                      (node) => node.id === item.sourceNodeId,
                    )?.title
                  }
                  {item.sourcePort && (
                    <span className="text-muted-foreground">
                      {" · "}
                      {canvasPortLabels[item.sourcePort] ?? item.sourcePort}
                    </span>
                  )}
                </span>
                <CanvasIconButton
                  size="icon-xs"
                  label="上移输入"
                  disabled={!editable || index === 0}
                  onClick={() => void move(index, -1)}
                >
                  <ArrowUp />
                </CanvasIconButton>
                <CanvasIconButton
                  size="icon-xs"
                  label="下移输入"
                  disabled={!editable || index === inputs.length - 1}
                  onClick={() => void move(index, 1)}
                >
                  <ArrowDown />
                </CanvasIconButton>
              </div>
            ))}
          </div>
        </section>
      )}
      {edge.output && <CanvasFixedOutput edge={edge} editable={editable} />}
      {error && (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      )}
      <div className="border-t border-foreground/[0.07] pt-3 dark:border-white/[0.08]">
        <CanvasButton
          variant="danger"
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
        </CanvasButton>
      </div>
    </div>
  );
}
