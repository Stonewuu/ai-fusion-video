"use client";

import { useRef, useState } from "react";
import {
  CircleAlert,
  Link2,
  Loader2,
  Play,
  RefreshCw,
  Save,
  X,
} from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { CanvasTextarea } from "../ui/canvas-field";
import { CanvasPanel } from "../ui/canvas-surface";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { userFacingError } from "@/lib/user-facing-error";
import { validateCanvasConfig } from "@/lib/canvas/config";
import {
  nodeConfigValueEqual,
  nodeDraftConflicts,
  rebaseNodeDraft,
} from "@/lib/canvas/node-draft";
import { canvasNodeConfigurationIssue } from "@/lib/canvas/node-readiness";
import { inputCountsForNode } from "@/lib/canvas/model-constraints";
import { runFinished, runLabels } from "@/lib/canvas/run-state";
import type { CanvasConfig, CanvasNode } from "@/lib/canvas/types";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import type { CanvasNodeStartResult } from "@/lib/canvas/node-execution";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import {
  useCanvasNodeActivity,
  useCanvasNodeRunReady,
} from "../runs/canvas-runs-provider";
import { useCanvasNodeDraft } from "../forms/use-node-draft";
import { useCanvasAssistantDraft } from "../agent/use-assistant-draft";
import { CanvasNodeConflicts } from "../forms/node-conflicts";
import { useGenerationInputs } from "./use-generation-inputs";
import {
  GenerationInputStrip,
  GenerationConnectedPrompt,
} from "./generation-inputs";
import { GenerationParameters } from "./generation-parameters";
import { GenerationInputsDialog } from "./generation-inputs-dialog";
import { TextGenerationWriteback } from "./text-writeback";
import { generationComposerCopy } from "@/lib/canvas/generation";

export function GenerationComposer({
  node,
  maxHeight,
  onClose,
  onPreview,
  onRun,
}: {
  node: CanvasNode;
  maxHeight: number;
  onClose: () => void;
  onPreview: (media: CanvasMediaPreview, returnFocus?: HTMLElement) => void;
  onRun: (nodeId: string) => Promise<CanvasNodeStartResult>;
}) {
  const store = useCanvasStoreApi();
  const snapshot = useCanvas((state) => state.snapshot!);
  const catalog = useCanvas((state) => state.catalog!);
  const pending = useCanvas(
    (state) => !!state.pending || state.saving || state.sync !== "connected",
  );
  const capability = catalog.nodes.find((item) => item.kind === node.kind)!;
  const { draft, setDraft, storageError, invalid, draftKey } =
    useCanvasNodeDraft(node);
  const inputs = useGenerationInputs(node.id);
  const inputIssues = [...new Map(inputs.value?.issues.map((issue) => [
    `${issue.nodeId}:${issue.message}`, issue,
  ])).values()];
  const activity = useCanvasNodeActivity(node.id);
  const ready = useCanvasNodeRunReady();
  const [busy, setBusy] = useState(false);
  const submitting = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [editingInputs, setEditingInputs] = useState(false);
  const config = draft?.config ?? node.config;
  const incoming = snapshot.graph.edges.filter(
    (edge) => edge.kind === "execution" && edge.targetNodeId === node.id,
  );
  const promptConnected = incoming.some((edge) => edge.targetPort === "prompt");
  const copy = generationComposerCopy(node.kind, promptConnected);
  const promptKey = copy.promptKey;
  const editable =
    snapshot.permissions.canEdit &&
    !snapshot.document.archived &&
    !pending &&
    !invalid &&
    !busy;
  const running = !!activity && !runFinished(activity.node.status);
  const waiting =
    !!activity &&
    [
      "WAITING_SELECTION",
      "WAITING_CONFIRMATION",
      "UNKNOWN_SUBMISSION",
      "NEEDS_REVIEW",
    ].includes(activity.node.status);
  const configurationIssue = canvasNodeConfigurationIssue(
    { ...node, config },
    capability,
    catalog,
    snapshot.graph,
  );
  const update = (next: CanvasConfig) => {
    setError(null);
    setDraft((current) => ({
      base: current?.base ?? node,
      title: current?.title ?? node.title,
      config: next,
    }));
  };
  const save = async () => {
    if (invalid) {
      throw new Error(storageError ?? "请先处理未保存的修改");
    }
    if (!draft) {
      return;
    }
    const current = store
      .getState()
      .snapshot!.graph.nodes.find((item) => item.id === node.id);
    if (!current) {
      throw new Error("节点已移除，请选择其他节点");
    }
    if (nodeDraftConflicts(draft, current).length) {
      throw new Error("此节点已被修改，请先确认要保留的内容");
    }
    const rebased = rebaseNodeDraft(draft, current, []);
    const validation = validateCanvasConfig(capability, rebased.config, false);
    if (validation) {
      throw new Error(validation);
    }
    const patch: Record<string, CanvasConfig[string] | null> = {};
    for (const key of new Set([
      ...Object.keys(current.config),
      ...Object.keys(rebased.config),
    ])) {
      if (!nodeConfigValueEqual(current.config[key], rebased.config[key])) {
        patch[key] = rebased.config[key] ?? null;
      }
    }
    if (Object.keys(patch).length || current.title !== rebased.title) {
      await store
        .getState()
        .edit("修改生成参数", [
          {
            type: "canvas.node.patch",
            nodeId: node.id,
            expectedConfigVersion: current.configVersion,
            configPatch: patch,
            title: rebased.title,
          },
        ]);
    }
    setDraft(null);
  };
  useCanvasAssistantDraft(draftKey, save);
  const submit = async (generate: boolean) => {
    if (submitting.current) {
      return;
    }
    submitting.current = true;
    setBusy(true);
    setError(null);
    try {
      await save();
      if (generate) {
        const result = await onRun(node.id);
        setError(result.error);
      }
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      submitting.current = false;
      setBusy(false);
    }
  };
  const canGenerate =
    editable &&
    snapshot.permissions.canExecute &&
    ready &&
    !running &&
    !inputs.loading &&
    !inputs.error &&
    !inputs.value?.issues.length &&
    !configurationIssue;
  return (
    <CanvasPanel
      aria-label={copy.name}
      data-canvas-generation-composer
      className="@container"
      style={{ maxHeight }}
    >
      <div className="flex shrink-0 items-center gap-1.5 px-3 pb-1 pt-2">
        <div className="min-w-0 flex-1">
          {inputs.value && (
            <GenerationInputStrip
              value={inputs.value}
              canEdit={editable}
              onPreview={onPreview}
              onFocus={(id) => store.getState().focus([id])}
              onEditInputs={() => setEditingInputs(true)}
              onError={setError}
            />
          )}
          {inputs.loading && (
            <p role="status" className="px-1 text-xs text-muted-foreground">
              读取输入…
            </p>
          )}
        </div>
        <CanvasIconButton
          label="连接素材"
          onClick={() => setEditingInputs(true)}
        >
          <Link2 />
        </CanvasIconButton>
        {draft && (
          <CanvasIconButton
            variant="secondary"
            label="保存参数"
            disabled={!editable}
            onClick={() => void submit(false)}
          >
            <Save />
          </CanvasIconButton>
        )}
        <CanvasIconButton label="收起" shortcut="Esc" onClick={onClose}>
          <X />
        </CanvasIconButton>
      </div>
      <div
        className="flex min-h-0 min-w-0 flex-col gap-2.5 overflow-y-auto px-3 pb-3 pt-1"
        style={{ maxHeight: Math.max(120, maxHeight - 120) }}
      >
        {inputs.error && (
          <div
            role="alert"
            className="flex items-center justify-between gap-2 rounded-[12px] border border-destructive/20 bg-destructive/5 px-3 py-2 text-xs text-destructive"
          >
            <p>{inputs.error}</p>
            <CanvasIconButton
              label="重新读取输入"
              variant="secondary"
              size="icon-sm"
              onClick={inputs.refresh}
            >
              <RefreshCw />
            </CanvasIconButton>
          </div>
        )}
        {inputs.value && (
          <GenerationConnectedPrompt prompt={inputs.value.prompt} />
        )}
        {inputIssues.map((issue) => (
          <div
            key={`${issue.nodeId}:${issue.message}`}
            role="alert"
            className="flex items-center justify-between gap-2 rounded-[12px] border border-destructive/20 bg-destructive/5 px-3 py-2 text-xs text-destructive"
          >
            <p>{issue.title}：{userFacingError(issue.message)}</p>
            <CanvasButton
              variant="secondary"
              size="sm"
              onClick={() => store.getState().focus([issue.nodeId])}
            >
              查看节点
            </CanvasButton>
          </div>
        ))}
        <div className="flex min-w-0 flex-col gap-2 py-0.5">
          <label htmlFor={`generation-prompt-${node.id}`} className="sr-only">
            {copy.promptLabel}
          </label>
          <CanvasTextarea
            id={`generation-prompt-${node.id}`}
            rows={3}
            disabled={!editable}
            className="h-24"
            placeholder={copy.placeholder}
            value={String(config[promptKey] ?? "")}
            maxLength={capability.configFields[promptKey]?.maxLength ?? 32000}
            onChange={(event) =>
              update({ ...config, [promptKey]: event.target.value })
            }
            onKeyDown={(event) => {
              if (
                (event.ctrlKey || event.metaKey) &&
                event.key === "Enter" &&
                canGenerate
              ) {
                event.preventDefault();
                void submit(true);
              }
            }}
          />
        </div>
        {node.kind === "textGeneration" && (
          <TextGenerationWriteback
            node={node}
            activity={activity}
            disabled={!editable}
          />
        )}
        {!promptConnected && config.additionalPrompt && (
          <div className="flex min-w-0 flex-col gap-2">
            <label
              htmlFor={`generation-additional-${node.id}`}
              className="text-xs font-medium text-muted-foreground"
            >
              补充要求
            </label>
            <CanvasTextarea
              id={`generation-additional-${node.id}`}
              rows={2}
              disabled={!editable}
              value={String(config.additionalPrompt)}
              maxLength={32000}
              onChange={(event) =>
                update({ ...config, additionalPrompt: event.target.value })
              }
            />
          </div>
        )}
        {draft && draft.base.configVersion !== node.configVersion && (
          <CanvasNodeConflicts
            draft={draft}
            current={node}
            onRebase={setDraft}
            disabled={pending || busy}
          />
        )}
        {storageError && (
          <p role="alert" className="text-sm text-destructive">
            {storageError}
          </p>
        )}
        {invalid && (
          <CanvasButton variant="secondary" onClick={() => setDraft(null)}>
            清理无效草稿
          </CanvasButton>
        )}
      </div>
      {(error || (!!config.modelId && configurationIssue)) && (
        <p role="alert" className="px-3 pb-2 text-xs text-destructive">
          {error ?? configurationIssue}
        </p>
      )}
      <footer className="flex shrink-0 flex-wrap items-center gap-2 border-t border-foreground/[0.07] bg-foreground/[0.02] px-3 py-2.5 dark:border-white/[0.08] dark:bg-white/[0.025]">
        <GenerationParameters
          capability={capability}
          config={config}
          disabled={!editable}
          connectedPorts={incoming.map((edge) => edge.targetPort!)}
          inputCounts={inputCountsForNode(node, snapshot.graph, catalog)}
          edited={promptKey}
          onChange={update}
        >
          <CanvasButton
            variant="generate"
            className="min-w-20"
            aria-label={copy.action}
            title={configurationIssue ?? "生成（Ctrl/⌘ Enter）"}
            disabled={!canGenerate}
            onClick={() => void submit(true)}
          >
            {waiting ? (
              <CircleAlert />
            ) : busy || running ? (
              <Loader2 className="animate-spin motion-reduce:animate-none" />
            ) : (
              <Play className="size-3 fill-current" />
            )}
            {waiting ? "待处理" : busy ? "提交中" : running ? "生成中" : "生成"}
          </CanvasButton>
        </GenerationParameters>
      </footer>
      <span className="sr-only" role="status">
        {busy
          ? "正在提交"
          : running
            ? runLabels[activity.node.status]
            : draft
              ? "有未保存的修改"
              : ""}
      </span>
      <GenerationInputsDialog
        open={editingInputs}
        node={node}
        capability={capability}
        disabled={!editable}
        unsaved={!!draft}
        onOpenChange={setEditingInputs}
        onSave={() => void submit(false)}
      />
    </CanvasPanel>
  );
}
