"use client";

import { useState, type ReactNode } from "react";
import { LocateFixed } from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import type { CanvasNodeRun, CanvasRun } from "@/lib/api/canvas-runs";
import { runLabels } from "@/lib/canvas/run-state";
import { canvasPortLabels } from "@/lib/canvas/labels";
import { GenerationExecutionDialog } from "@/components/dashboard/generation/generation-execution-dialog";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
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
import { cn } from "@/lib/utils";

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
  const statusTone =
    node.status === "SUCCEEDED"
      ? "bg-emerald-500/12 text-emerald-700 dark:text-emerald-300"
      : ["FAILED", "PARTIAL_FAILED", "NEEDS_REVIEW", "UNKNOWN_SUBMISSION"].includes(
            node.status,
          )
        ? "bg-destructive/10 text-destructive"
        : "bg-foreground/[0.06] text-muted-foreground dark:bg-white/[0.08]";

  return (
    <section className="flex flex-col gap-2.5 rounded-[12px] border border-foreground/[0.07] bg-background/70 p-3 dark:border-white/[0.08] dark:bg-white/[0.03]">
      <div className="flex items-center justify-between gap-2">
        {node.parentNodeRunId ? (
          <span className="min-w-0 truncate text-[13px] font-medium">
            {title}
          </span>
        ) : (
          <CanvasButton
            variant="ghost"
            size="sm"
            className="min-w-0 justify-start px-1.5"
            onClick={() => store.getState().focus([node.nodeId])}
          >
            <LocateFixed className="size-3.5 text-muted-foreground" />
            <span className="truncate">{title}</span>
          </CanvasButton>
        )}
        <span
          className={cn(
            "shrink-0 rounded-[6px] px-1.5 py-0.5 text-[11px] font-medium",
            statusTone,
          )}
        >
          {runLabels[node.status]}
        </span>
      </div>
      {!node.parentNodeRunId && node.derivedState !== "CURRENT" && (
        <p className="text-[11px] text-muted-foreground">
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
      <div className="flex flex-wrap items-center gap-1.5 empty:hidden">
        {node.externalKind === "GENERATION" && node.externalId && (
          <CanvasButton
            size="sm"
            variant="secondary"
            onClick={() => setExecution(true)}
          >
            查看生成详情
          </CanvasButton>
        )}
        {node.externalKind === "MEDIA" && node.externalId && (
          <CanvasButton
            size="sm"
            variant="secondary"
            onClick={() => setMedia(true)}
          >
            查看媒体处理
          </CanvasButton>
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
      </div>
      {node.status === "WAITING_SELECTION" && !node.contentUnavailable && (
        <div className="flex flex-col gap-2.5">
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
          className="flex flex-col gap-2"
        >
          <summary className="cursor-pointer rounded-[6px] text-xs text-muted-foreground outline-none focus-visible:ring-2 focus-visible:ring-primary/30">
            {outputs.length} 个结果
          </summary>
          <div className="flex flex-col gap-2 pt-2">
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
            <div className="flex items-center justify-between gap-2 pt-1">
              <CanvasButton
                size="sm"
                variant="ghost"
                disabled={current === 0}
                onClick={() => setPage(current - 1)}
              >
                上一页
              </CanvasButton>
              <span className="text-[11px] tabular-nums text-muted-foreground">
                {current + 1} / {pages}
              </span>
              <CanvasButton
                size="sm"
                variant="ghost"
                disabled={current + 1 === pages}
                onClick={() => setPage(current + 1)}
              >
                下一页
              </CanvasButton>
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
          <div>
            <CanvasButton
              size="sm"
              variant="ghost"
              onClick={() => onContinue(node.nodeId, node.id)}
            >
              用此结果继续
            </CanvasButton>
          </div>
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
