"use client";

import { useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { CanvasNodeRun, CanvasRun } from "@/lib/api/canvas-runs";
import { runLabels } from "@/lib/canvas/run-state";
import { canvasPortLabels } from "@/lib/canvas/labels";
import { GenerationExecutionDialog } from "@/components/dashboard/generation/generation-execution-dialog";
import { useCanvasStoreApi } from "../canvas-provider";
import { useCanvasRuns } from "./canvas-runs-provider";
import { CanvasRunResult } from "./run-result";
import { CanvasMediaExecutionDialog } from "./media-execution-dialog";
import { CanvasTextExecutionButton } from "./text-execution-dialog";
import { CanvasWritebackDialog } from "./writeback-dialog";
import { CanvasAgentExecutionButton } from "../agent/execution-dialog";
import { CanvasMcpExecutionButton } from "./mcp-execution-dialog";
import { CanvasExportDownloadButton } from "../delivery/export-history";
import { userFacingError } from "@/lib/user-facing-error";
import { CanvasGuidance } from "../shared/canvas-guidance";
import { useCanvas } from "../canvas-provider";

export function CanvasRunNode({
  run,
  node,
  onContinue,
  children,
}: {
  run: CanvasRun;
  node: CanvasNodeRun;
  onContinue?: (nodeId: string, nodeRunId: string) => void;
  children?: ReactNode;
}) {
  const runtime = useCanvasRuns();
  const store = useCanvasStoreApi();
  const [execution, setExecution] = useState(false);
  const [media, setMedia] = useState(false);
  const [page, setPage] = useState(0);
  const hasNextStep = useCanvas(
    (state) =>
      state.snapshot?.graph.edges.some(
        (edge) =>
          edge.kind === "execution" && edge.sourceNodeId === node.nodeId,
      ) === true,
  );
  const outputs = Object.entries(node.outputs.values).flatMap(
    ([port, values]) => values.map((value, index) => ({ port, value, index })),
  );
  const pages = Math.max(1, Math.ceil(outputs.length / 8));
  const current = Math.min(page, pages - 1);
  const title = node.parentNodeRunId
    ? `第 ${node.itemIndex + 1} 项 · 第 ${node.attempt} 次执行`
    : node.title;
  return (
    <section className="space-y-3 border-t border-border/20 pt-4">
      <div className="flex items-center justify-between gap-2">
        {node.parentNodeRunId ? (
          <span className="text-sm font-medium">{title}</span>
        ) : (
          <Button
            variant="link"
            size="sm"
            onClick={() => store.getState().focus([node.nodeId])}
          >
            {title}
          </Button>
        )}
        <Badge variant="outline">{runLabels[node.status]}</Badge>
      </div>
      {!node.parentNodeRunId && node.derivedState !== "CURRENT" && (
        <p className="text-xs text-muted-foreground">
          {node.derivedState === "NODE_DELETED"
            ? "节点已从画布移除，保留此次结果"
            : node.derivedState === "SOURCE_UNAVAILABLE"
              ? "原输入目前不可用"
              : "输入已有修改，此处保留运行时的结果"}
        </p>
      )}
      {node.error && (
        <CanvasGuidance title="这一步未完成" tone="error">
          {userFacingError(node.error)}
        </CanvasGuidance>
      )}
      {node.externalKind === "GENERATION" && node.externalId && (
        <Button size="xs" variant="outline" onClick={() => setExecution(true)}>
          查看生成详情
        </Button>
      )}
      {node.externalKind === "MEDIA" && node.externalId && (
        <Button size="xs" variant="outline" onClick={() => setMedia(true)}>
          查看媒体处理
        </Button>
      )}
      {node.externalKind === "TEXT" && node.externalId && (
        <CanvasTextExecutionButton runId={run.id} nodeId={node.id} />
      )}
      {node.externalKind === "EXPORT" &&
        node.externalId &&
        node.status === "SUCCEEDED" &&
        !node.contentUnavailable && (
          <CanvasExportDownloadButton
            canvasId={run.canvasId}
            exportId={node.externalId}
          />
        )}
      {node.externalKind === "MCP" &&
      node.externalId &&
      !node.contentUnavailable ? (
        <CanvasMcpExecutionButton runId={run.id} nodeId={node.id} />
      ) : null}
      {node.externalKind === "AGENT" &&
      node.externalId &&
      !node.contentUnavailable ? (
        <CanvasAgentExecutionButton
          runId={run.id}
          nodeId={node.id}
          attention={node.status === "WAITING_CONFIRMATION" && run.canControl}
          interrupted={node.status === "UNKNOWN_SUBMISSION"}
        />
      ) : null}
      {node.status === "WAITING_SELECTION" && !node.contentUnavailable && (
        <div className="space-y-3">
          <CanvasGuidance title="请选择一个结果" tone="action">
            选定后，后续步骤会使用这份结果继续运行。
          </CanvasGuidance>
          {node.inputs?.values.candidates.map((value, index) => (
            <CanvasRunResult
              key={value.id}
              value={value}
              title={`${title} · 候选 ${index + 1}`}
              sourceNodeId={node.nodeId}
              disabled={runtime.busy || !run.canControl || run.cancelRequested}
              onSelect={() => void runtime.select(node, value.id)}
            />
          ))}
        </div>
      )}
      {node.writeback && <CanvasWritebackDialog run={run} node={node} />}
      {children}
      {!!outputs.length && (
        <details
          open={
            node.status === "SUCCEEDED" &&
            outputs.length <= 2 &&
            ![
              "text",
              "story",
              "collection",
              "image",
              "video",
              "audio",
            ].includes(node.kind)
          }
          className="space-y-2"
        >
          <summary className="cursor-pointer text-xs text-muted-foreground">
            {outputs.length} 个结果
          </summary>
          <div className="space-y-2">
            {outputs
              .slice(current * 8, current * 8 + 8)
              .map(({ value, port, index }) => (
                <CanvasRunResult
                  key={`${port}:${value.id}:${index}`}
                  value={value}
                  title={`${title} · ${canvasPortLabels[port] ?? port} ${index + 1}`}
                  sourceNodeId={node.nodeId}
                />
              ))}
          </div>
          {pages > 1 && (
            <div className="flex items-center justify-between gap-2">
              <Button
                size="xs"
                variant="ghost"
                disabled={current === 0}
                onClick={() => setPage(current - 1)}
              >
                上一页
              </Button>
              <span className="text-xs text-muted-foreground">
                {current + 1} / {pages}
              </span>
              <Button
                size="xs"
                variant="ghost"
                disabled={current + 1 === pages}
                onClick={() => setPage(current + 1)}
              >
                下一页
              </Button>
            </div>
          )}
        </details>
      )}
      {!node.parentNodeRunId &&
        node.status === "SUCCEEDED" &&
        !node.contentUnavailable &&
        node.derivedState !== "NODE_DELETED" &&
        hasNextStep &&
        onContinue && (
          <Button
            size="xs"
            variant="ghost"
            onClick={() => onContinue(node.nodeId, node.id)}
          >
            用此结果继续
          </Button>
        )}
      {execution && node.externalId && (
        <GenerationExecutionDialog
          id={node.externalId}
          onClose={() => setExecution(false)}
          onChanged={runtime.reload}
        />
      )}
      {media && !node.contentUnavailable && (
        <CanvasMediaExecutionDialog
          runId={run.id}
          nodeId={node.id}
          onClose={() => setMedia(false)}
        />
      )}
    </section>
  );
}
