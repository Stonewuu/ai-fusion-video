"use client";

import { useRef, useState, type CSSProperties } from "react";
import {
  Check,
  CircleAlert,
  Link2,
  Loader2,
  Play,
  Replace,
  X,
} from "lucide-react";
import {
  Dialog,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { CanvasDialogContent } from "@/components/canvas/shared/canvas-dialog";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { CanvasInput } from "../ui/canvas-field";
import { CanvasPanel } from "../ui/canvas-surface";
import { CanvasNodeTypeIcon } from "../shared/canvas-node-icon";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { validateCanvasConfig } from "@/lib/canvas/config";
import { inputCountsForNode } from "@/lib/canvas/model-constraints";
import {
  nodeConfigValueEqual,
  nodeDraftConflicts,
  rebaseNodeDraft,
} from "@/lib/canvas/node-draft";
import {
  canRunCanvasNode,
  type CanvasNodeStartResult,
} from "@/lib/canvas/node-execution";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import { canvasNodeConfigurationIssue } from "@/lib/canvas/node-readiness";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { runFinished, runLabels } from "@/lib/canvas/run-state";
import type { CanvasConfig, CanvasNode } from "@/lib/canvas/types";
import {
  useCanvas,
  useCanvasResource,
  useCanvasStoreApi,
} from "../canvas-provider";
import {
  useCanvasNodeActivity,
  useCanvasNodeRunReady,
} from "../runs/canvas-runs-provider";
import { useCanvasNodeDraft } from "../forms/use-node-draft";
import { useCanvasAssistantDraft } from "../agent/use-assistant-draft";
import { CanvasNodeConflicts } from "../forms/node-conflicts";
import { CanvasConfigFields } from "../forms/config-fields";
import { CanvasResourcePicker } from "../forms/resource-picker";
import { CanvasResourceEditor } from "../forms/resource-editor";
import { CanvasResourceMedia } from "../inspector/resource-media";
import { CanvasResourceActions } from "../inspector/resource-actions";
import { GenerationInputsDialog } from "./generation-inputs-dialog";

/**
 * 通用节点悬浮面板：承载配置类节点、运行步骤、分组及项目内容引用节点的查看、编辑与运行。
 * 统一使用画布悬浮外壳与单行底栏，不再在点击节点时弹出右侧属性栏挤压画布。
 */
export function NodeFloatingPanel({
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
  const resource = useCanvasResource(node.resource);
  const { draft, setDraft, storageError, invalid, draftKey } =
    useCanvasNodeDraft(node);
  const activity = useCanvasNodeActivity(node.id);
  const ready = useCanvasNodeRunReady();
  const [busy, setBusy] = useState(false);
  const submitting = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [editingInputs, setEditingInputs] = useState(false);
  const [rebinding, setRebinding] = useState(false);
  const config = draft?.config ?? node.config;
  const title = draft?.title ?? node.title;
  const incoming = snapshot.graph.edges.filter(
    (edge) => edge.kind === "execution" && edge.targetNodeId === node.id,
  );
  const editable =
    snapshot.permissions.canEdit &&
    !snapshot.document.archived &&
    !pending &&
    !invalid &&
    !busy;
  const runnable = canRunCanvasNode(capability);
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
  const updateConfig = (next: CanvasConfig) => {
    setError(null);
    setDraft((current) => ({
      base: current?.base ?? node,
      title: current?.title ?? node.title,
      config: next,
    }));
  };
  const updateTitle = (next: string) => {
    setError(null);
    setDraft((current) => ({
      base: current?.base ?? node,
      title: next,
      config: current?.config ?? node.config,
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
    if (!rebased.title.trim()) {
      throw new Error("节点名称不能为空");
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
    if (Object.keys(patch).length || current.title !== rebased.title.trim()) {
      await store.getState().edit("修改节点配置", [
        {
          type: "canvas.node.patch",
          nodeId: node.id,
          expectedConfigVersion: current.configVersion,
          title:
            current.title !== rebased.title.trim()
              ? rebased.title.trim()
              : undefined,
          configPatch: patch,
        },
      ]);
    }
    setDraft(null);
  };
  useCanvasAssistantDraft(draftKey, save);
  const submit = async (execute: boolean) => {
    if (submitting.current) {
      return;
    }
    submitting.current = true;
    setBusy(true);
    setError(null);
    try {
      await save();
      if (execute) {
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
  const hasConfigFields = Object.keys(capability.configFields).length > 0;
  const hasConflict = Boolean(
    draft && draft.base.configVersion !== node.configVersion,
  );
  const hasBody =
    hasConfigFields ||
    Boolean(capability.resourceType) ||
    Boolean(configurationIssue) ||
    hasConflict ||
    Boolean(storageError) ||
    invalid ||
    Boolean(error);
  const canRun =
    runnable &&
    editable &&
    snapshot.permissions.canExecute &&
    ready &&
    !running &&
    !configurationIssue;
  return (
    <CanvasPanel
      aria-label={`编辑${capability.label}`}
      data-canvas-generation-composer
      className="@container"
      style={
        {
          maxHeight,
          "--canvas-tone": canvasNodeColor(node.kind),
        } as CSSProperties
      }
    >
      <div className="flex shrink-0 items-center gap-2 px-3 py-2.5">
        <span className="canvas-node-badge-icon shrink-0">
          <CanvasNodeTypeIcon
            capability={capability}
            className="size-3.5"
            aria-hidden
          />
        </span>
        <div className="min-w-0 flex-1">
          <CanvasInput
            aria-label={node.resource ? "画布显示名称" : "节点名称"}
            placeholder="节点名称"
            value={title}
            maxLength={200}
            disabled={!editable}
            onChange={(event) => updateTitle(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.nativeEvent.isComposing) {
                event.preventDefault();
                void submit(false);
              }
            }}
          />
        </div>
        {title.trim() !== capability.label && (
          <span className="hidden shrink-0 text-xs text-muted-foreground @[440px]:inline">
            {capability.label}
          </span>
        )}
        {draft && (
          <span
            role="status"
            className="inline-flex shrink-0 items-center gap-1 text-xs text-muted-foreground"
          >
            <span className="size-1.5 rounded-full bg-amber-500" aria-hidden />
            未保存
          </span>
        )}
        {capability.inputs.length > 0 && (
          <CanvasIconButton
            label="输入来源"
            hint={
              incoming.length
                ? `输入来源（已连 ${incoming.length} 条）`
                : "连接输入来源"
            }
            onClick={() => setEditingInputs(true)}
          >
            <Link2 />
          </CanvasIconButton>
        )}
        {capability.resourceType && node.resource && (
          <CanvasIconButton
            label="更换引用"
            disabled={!editable}
            onClick={() => setRebinding(true)}
          >
            <Replace />
          </CanvasIconButton>
        )}
        <CanvasIconButton label="收起" shortcut="Esc" onClick={onClose}>
          <X />
        </CanvasIconButton>
      </div>
      {hasBody && (
        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto border-t border-foreground/[0.07] px-3 py-2.5 dark:border-white/[0.08]">
          <div className="flex min-w-0 flex-col gap-4 p-1">
            {configurationIssue && (
              <p
                role="status"
                className="rounded-[12px] border border-amber-500/25 bg-amber-500/8 px-3 py-2 text-xs text-amber-800 dark:text-amber-200"
              >
                {configurationIssue}
              </p>
            )}
            {capability.resourceType && !node.resource && (
              <CanvasResourcePicker
                type={capability.resourceType}
                onSelect={(entry) => {
                  void store
                    .getState()
                    .edit("关联项目内容", [
                      {
                        type: "canvas.node.bind",
                        nodeId: node.id,
                        expectedConfigVersion: node.configVersion,
                        resource: entry.ref,
                      },
                    ])
                    .catch((failure) => setError(getApiErrorMessage(failure)));
                }}
              />
            )}
            {resource.loading && !resource.data && (
              <p role="status" className="text-xs text-muted-foreground">
                正在读取项目内容…
              </p>
            )}
            {resource.error && (
              <p role="alert" className="text-xs text-destructive">
                {resource.error}
              </p>
            )}
            {resource.data && (
              <div className="overflow-hidden rounded-[12px] border border-foreground/[0.07] bg-background/60 dark:border-white/[0.08]">
                <CanvasResourceMedia
                  resource={resource.data}
                  onPreview={onPreview}
                />
                <CanvasResourceEditor
                  key={`${resource.data.header.resourceType}:${resource.data.header.resourceId}`}
                  resource={resource.data}
                  onPreview={onPreview}
                />
                <CanvasResourceActions
                  key={`actions:${resource.data.header.resourceType}:${resource.data.header.resourceId}`}
                  resource={resource.data}
                />
              </div>
            )}
            {hasConfigFields && (
              <CanvasConfigFields
                capability={capability}
                config={config}
                disabled={!editable}
                connectedPorts={incoming.map((edge) => edge.targetPort!)}
                inputCounts={inputCountsForNode(node, snapshot.graph, catalog)}
                onChange={updateConfig}
              />
            )}
            {hasConflict && draft && (
              <CanvasNodeConflicts
                key={node.configVersion}
                draft={draft}
                current={node}
                onRebase={setDraft}
                disabled={pending || busy}
              />
            )}
            {storageError && (
              <p role="alert" className="text-xs text-destructive">
                {storageError}
              </p>
            )}
            {invalid && (
              <CanvasButton variant="secondary" onClick={() => setDraft(null)}>
                清理无效草稿
              </CanvasButton>
            )}
            {error && (
              <p role="alert" className="text-xs text-destructive">
                {error}
              </p>
            )}
          </div>
        </div>
      )}
      <footer className="flex shrink-0 flex-wrap items-center gap-2 border-t border-foreground/[0.07] bg-foreground/[0.02] px-3 py-2.5 dark:border-white/[0.08] dark:bg-white/[0.025]">
        {draft ? (
          <CanvasButton
            variant="ghost"
            disabled={pending || busy}
            onClick={() => {
              setDraft(null);
              setError(null);
            }}
          >
            放弃修改
          </CanvasButton>
        ) : running && activity ? (
          <span className="pl-0.5 text-xs text-muted-foreground">
            {runLabels[activity.node.status]}
          </span>
        ) : node.kind === "group" ? (
          <span className="pl-0.5 text-xs text-muted-foreground">
            拖入节点归组，双击标题也可改名
          </span>
        ) : null}
        <div className="ml-auto flex items-center gap-2">
          {draft && (
            <CanvasButton
              variant={runnable ? "secondary" : "generate"}
              className="min-w-20"
              disabled={!editable || hasConflict}
              onClick={() => void submit(false)}
            >
              {busy ? (
                <Loader2 className="animate-spin motion-reduce:animate-none" />
              ) : (
                <Check className="size-3.5" />
              )}
              保存
            </CanvasButton>
          )}
          {runnable ? (
            <CanvasButton
              variant="generate"
              className="min-w-20"
              disabled={!canRun}
              title={configurationIssue ?? "运行此节点"}
              onClick={() => void submit(true)}
            >
              {waiting ? (
                <CircleAlert />
              ) : busy || running ? (
                <Loader2 className="animate-spin motion-reduce:animate-none" />
              ) : (
                <Play className="size-3 fill-current" />
              )}
              {waiting
                ? "待处理"
                : busy
                  ? "提交中"
                  : running
                    ? "运行中"
                    : "运行"}
            </CanvasButton>
          ) : !draft ? (
            <CanvasButton
              variant="secondary"
              className="min-w-20"
              onClick={onClose}
            >
              完成
            </CanvasButton>
          ) : null}
        </div>
      </footer>
      {capability.inputs.length > 0 && (
        <GenerationInputsDialog
          open={editingInputs}
          node={node}
          capability={capability}
          disabled={!editable}
          unsaved={!!draft}
          onOpenChange={setEditingInputs}
          onSave={() => void submit(false)}
        />
      )}
      {capability.resourceType && (
        <Dialog open={rebinding} onOpenChange={setRebinding}>
          <CanvasDialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden">
            <DialogHeader className="shrink-0">
              <DialogTitle>更换内容引用</DialogTitle>
              <DialogDescription>
                更换当前节点引用，其他引用节点保持各自绑定。
              </DialogDescription>
            </DialogHeader>
            <DialogScrollBody>
              <CanvasResourcePicker
                type={capability.resourceType}
                value={node.resource?.id}
                onSelect={(entry) => {
                  void store
                    .getState()
                    .edit("更换节点引用", [
                      {
                        type: "canvas.node.bind",
                        nodeId: node.id,
                        expectedConfigVersion: node.configVersion,
                        resource: entry.ref,
                      },
                    ])
                    .then(() => setRebinding(false))
                    .catch((failure) => setError(getApiErrorMessage(failure)));
                }}
              />
            </DialogScrollBody>
          </CanvasDialogContent>
        </Dialog>
      )}
    </CanvasPanel>
  );
}
