"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent } from "../shared/canvas-dialog";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { getApiErrorMessage } from "@/lib/api/api-error";
import {
  canvasInputCandidates,
  canvasInputPurposes,
  connectCanvasInput,
} from "@/lib/canvas/input-bindings";
import { canvasPortLabels, canvasTypeNames } from "@/lib/canvas/labels";
import type { CanvasGraph } from "@/lib/canvas/types";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";

function inputSignature(graph: CanvasGraph, nodeId: string, portId: string) {
  return JSON.stringify(
    graph.edges
      .filter(
        (edge) =>
          edge.kind === "execution" &&
          edge.targetNodeId === nodeId &&
          edge.targetPort === portId,
      )
      .map((edge) => [edge.id, edge.version])
      .sort((a, b) => String(a[0]).localeCompare(String(b[0]))),
  );
}

export function CanvasInputPicker({
  nodeId,
  portId,
  onClose,
}: {
  nodeId: string;
  portId: string;
  onClose: () => void;
}) {
  const store = useCanvasStoreApi();
  const snapshot = useCanvas((state) => state.snapshot!);
  const catalog = useCanvas((state) => state.catalog!);
  const pending = useCanvas((state) => !!state.pending);
  const target = snapshot.graph.nodes.find((node) => node.id === nodeId);
  const bindingSignature = inputSignature(snapshot.graph, nodeId, portId);
  const [baseline] = useState(() => ({
    version: target?.configVersion,
    bindings: bindingSignature,
  }));
  const [query, setQuery] = useState("");
  const [choice, setChoice] = useState<string | null>(null);
  const [sourceVersion, setSourceVersion] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const input = catalog.nodes
    .find((item) => item.kind === target?.kind)
    ?.inputs.find((port) => port.id === portId);
  const candidates =
    target && input
      ? canvasInputCandidates(snapshot.graph, catalog.nodes, target, input)
      : [];
  const shown = candidates.filter((item) =>
    `${item.node.title} ${item.port.label}`
      .toLocaleLowerCase()
      .includes(query.toLocaleLowerCase()),
  );
  const selected = candidates.find(
    (item) => `${item.node.id}:${item.port.id}` === choice,
  );
  const changed =
    baseline.version !== target?.configVersion ||
    baseline.bindings !== bindingSignature;
  const sourceChanged =
    !!selected && selected.node.configVersion !== sourceVersion;
  const connect = async () => {
    if (!selected || busy) return;
    setBusy(true);
    try {
      const state = store.getState();
      const current = state.snapshot!.graph;
      if (
        current.nodes.find((node) => node.id === nodeId)?.configVersion !==
          baseline.version ||
        inputSignature(current, nodeId, portId) !== baseline.bindings ||
        current.nodes.find((node) => node.id === selected.node.id)
          ?.configVersion !== sourceVersion
      )
        throw new Error("节点或输入已有修改，请重新选择来源");
      const operations = connectCanvasInput(
        state.snapshot!.graph,
        state.catalog!.nodes,
        nodeId,
        portId,
        selected.node.id,
        selected.port.id,
        crypto.randomUUID(),
      );
      await state.edit("连接创作输入", operations);
      onClose();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <CanvasDialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>
            选择{canvasPortLabels[portId] ?? input?.label}来源
          </DialogTitle>
          <DialogDescription>
            {canvasInputPurposes[portId] ??
              "选择画布中的内容，运行时会使用此来源的输出。"}
          </DialogDescription>
        </DialogHeader>
        <Input
          aria-label="搜索输入来源"
          placeholder="搜索节点名称"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <DialogScrollBody className="flex flex-col gap-3">
          {target && Object.hasOwn(target.config, portId) && (
            <div className="flex flex-col gap-2 text-sm text-muted-foreground">
              <p>连接后替换以下手填内容：</p>
              <p className="line-clamp-4 whitespace-pre-wrap">
                {String(target.config[portId])}
              </p>
            </div>
          )}
          {changed && (
            <p role="alert" className="text-sm text-destructive">
              节点或输入连接已有修改，请关闭后重新选择。
            </p>
          )}
          {sourceChanged && (
            <p role="alert" className="text-sm text-destructive">
              所选来源已有修改，请重新选择以使用当前内容。
            </p>
          )}
          {!shown.length && (
            <p className="text-sm text-muted-foreground">
              没有匹配的来源。可先从项目内容或素材库放入画布。
            </p>
          )}
          <div className="flex flex-col gap-2">
            {shown.slice(0, 80).map((item) => {
              const key = `${item.node.id}:${item.port.id}`;
              return (
                <Button
                  key={key}
                  variant={choice === key ? "secondary" : "ghost"}
                  className="w-full justify-start"
                  aria-pressed={choice === key}
                  disabled={!!item.issue || busy || changed}
                  title={item.issue ?? undefined}
                  onClick={() => {
                    setChoice(key);
                    setSourceVersion(item.node.configVersion);
                  }}
                >
                  <span className="min-w-0 flex-1 truncate text-left">
                    {item.node.title} ·{" "}
                    {canvasPortLabels[item.port.id] ?? item.port.label}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {canvasTypeNames[item.type]}
                    {item.port.collection ? "集合" : ""}
                  </span>
                </Button>
              );
            })}
          </div>
          {shown.length > 80 && (
            <p className="text-xs text-muted-foreground">
              共 {shown.length} 个来源，请输入名称缩小范围。
            </p>
          )}
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </DialogScrollBody>
        <DialogFooter className="shrink-0">
          <Button variant="outline" disabled={busy} onClick={onClose}>
            取消
          </Button>
          <Button
            disabled={
              busy ||
              pending ||
              changed ||
              sourceChanged ||
              !snapshot.permissions.canEdit ||
              snapshot.document.archived ||
              !selected ||
              !!selected.issue
            }
            onClick={() => void connect()}
          >
            {busy ? "连接中…" : "使用此来源"}
          </Button>
        </DialogFooter>
      </CanvasDialogContent>
    </Dialog>
  );
}
