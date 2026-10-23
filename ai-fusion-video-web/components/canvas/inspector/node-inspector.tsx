"use client";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";

import { useState } from "react";
import { Save, Link2, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { validateCanvasConfig } from "@/lib/canvas/config";
import { nodeMedia, type CanvasMediaPreview } from "@/lib/canvas/presentation";
import { useCanvasNodeDraft } from "../forms/use-node-draft";
import { nodeConfigValueEqual } from "@/lib/canvas/node-draft";
import { useCanvasAssistantDraft } from "../agent/use-assistant-draft";
import { CanvasNodeConflicts } from "../forms/node-conflicts";
import { inputCountsForNode } from "@/lib/canvas/model-constraints";
import { canvasNodeConfigurationIssue } from "@/lib/canvas/node-readiness";
import { CanvasGuidance } from "../shared/canvas-guidance";
import type { CanvasConfig, CanvasNode } from "@/lib/canvas/types";
import {
  useCanvas,
  useCanvasResource,
  useCanvasStoreApi,
} from "../canvas-provider";
import { CanvasConfigFields } from "../forms/config-fields";
import { CanvasResourcePicker } from "../forms/resource-picker";
import { CanvasResourceEditor } from "../forms/resource-editor";
import { canvasMediaKinds } from "@/lib/canvas/placement";
import { CanvasMediaPicker } from "../library/media-picker";
import { CanvasResourceMedia } from "./resource-media";
import { CanvasResourceActions } from "./resource-actions";
import { CanvasNodeInputs } from "./node-inputs";

export function CanvasNodeInspector({
  nodeId,
  onPreview,
}: {
  nodeId: string;
  onPreview: (media: CanvasMediaPreview) => void;
}) {
  const snapshot = useCanvas((state) => state.snapshot!);
  const node = snapshot.graph.nodes.find((item) => item.id === nodeId);
  if (!node)
    return (
      <p className="p-4 text-sm text-muted-foreground">
        该节点已移除，请选择其他节点。
      </p>
    );
  return <CanvasNodeDetails key={node.id} node={node} onPreview={onPreview} />;
}

function CanvasNodeDetails({
  node,
  onPreview,
}: {
  node: CanvasNode;
  onPreview: (media: CanvasMediaPreview) => void;
}) {
  const store = useCanvasStoreApi();
  const capability = useCanvas(
    (state) => state.catalog!.nodes.find((item) => item.kind === node.kind)!,
  );
  const snapshot = useCanvas((state) => state.snapshot!);
  const pending = useCanvas((state) => !!state.pending);
  const resource = useCanvasResource(node.resource);
  const { draft, setDraft, storageError, invalid, draftKey } =
    useCanvasNodeDraft(node);
  const [error, setError] = useState<string | null>(null);
  const [rebinding, setRebinding] = useState(false);
  const editable =
    snapshot.permissions.canEdit &&
    !snapshot.document.archived &&
    !pending &&
    !invalid;
  const config = draft?.config ?? node.config;
  const title = draft?.title ?? node.title;
  const incoming = snapshot.graph.edges.filter(
    (edge) => edge.kind === "execution" && edge.targetNodeId === node.id,
  );
  const mediaKind =
    canvasMediaKinds[node.kind as keyof typeof canvasMediaKinds];
  const preview = nodeMedia(node, resource.data);
  const configurationIssue = canvasNodeConfigurationIssue(
    { ...node, config },
    capability,
    store.getState().catalog!,
    snapshot.graph,
  );
  const changes = () => {
    const base = draft?.base ?? node;
    const patch: Record<string, CanvasConfig[string] | null> = {};
    for (const key of new Set([
      ...Object.keys(base.config),
      ...Object.keys(config),
    ])) {
      if (!nodeConfigValueEqual(base.config[key], config[key])) patch[key] = config[key] ?? null;
    }
    return { patch, title: base.title !== title ? title : undefined };
  };
  const save = async () => {
    const validation = validateCanvasConfig(capability, config, false);
    if (validation) {
      setError(validation);
      return;
    }
    if (!title.trim()) {
      setError("节点名称不能为空");
      return;
    }
    const changed = changes();
    if (!Object.keys(changed.patch).length && changed.title === undefined) {
      setDraft(null);
      return;
    }
    try {
      await store.getState().edit("修改节点配置", [
        {
          type: "canvas.node.patch",
          nodeId: node.id,
          expectedConfigVersion:
            draft?.base.configVersion ?? node.configVersion,
          title: changed.title,
          configPatch: changed.patch,
        },
      ]);
      setDraft(null);
      setError(null);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    }
  };
  useCanvasAssistantDraft(draftKey, async () => {
    if (invalid) throw new Error(storageError ?? "请先处理节点草稿错误。");
    if (draft) await save();
  });
  const settings = (
    <section className="space-y-4 border-b border-border/20 p-4">
      {draft && (
        <div className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-2 border-b border-border/20 bg-card py-3">
          <span className="text-xs text-muted-foreground" role="status">节点配置未保存</span>
          <div className="flex gap-2">
          <Button
            size="sm"
            disabled={
              !editable || draft.base.configVersion !== node.configVersion
            }
            onClick={() => void save()}
          >
            <Save />
            保存节点
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={pending}
            onClick={() => {
              setDraft(null);
              setError(null);
            }}
          >
            放弃修改
          </Button>
          </div>
        </div>
      )}
      {configurationIssue && (
        <CanvasGuidance title="运行前需要调整" tone="warning">
          {configurationIssue}。
        </CanvasGuidance>
      )}
      <div className="flex flex-col gap-2">
        <p className="text-xs text-muted-foreground">{capability.label}设置</p>
        <Input
          aria-label={node.resource ? "画布显示名称" : "节点名称"}
          value={title}
          maxLength={200}
          disabled={!editable}
          onChange={(event) =>
            setDraft((current) => ({
              base: current?.base ?? node,
              title: event.target.value,
              config: current?.config ?? node.config,
            }))
          }
        />
      </div>
      {preview && (
        <Button variant="outline" size="sm" onClick={() => onPreview(preview)}>
          <Eye />
          预览媒体
        </Button>
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
      {capability.resourceType && node.resource && (
        <>
          <p className="text-xs text-muted-foreground">
            显示名称只用于当前画布。下方可以更换关联的项目内容。
          </p>
          <Button
            variant="ghost"
            size="sm"
            disabled={!editable}
            onClick={() => setRebinding(true)}
          >
            <Link2 />
            更换引用
          </Button>
        </>
      )}
      {mediaKind && (
        <details open={!config.resourceId || undefined}>
          <summary className="cursor-pointer text-sm font-medium">
            {config.resourceId ? "更换媒体" : "选择或上传媒体"}
          </summary>
          <div className="pt-3">
            <CanvasMediaPicker
              kind={mediaKind}
              disabled={!editable}
              value={String(config.resourceId ?? "")}
              onSelect={(media) =>
                setDraft((current) => ({
                  base: current?.base ?? node,
                  title: current?.title ?? node.title,
                  config: {
                    ...(current?.config ?? node.config),
                    resourceId: media.id,
                  },
                }))
              }
            />
          </div>
        </details>
      )}
      <CanvasConfigFields
        capability={capability}
        config={config}
        disabled={!editable}
        connectedPorts={incoming.map((edge) => edge.targetPort!)}
        inputCounts={inputCountsForNode(
          node,
          snapshot.graph,
          store.getState().catalog!,
        )}
        onChange={(next) =>
          setDraft((current) => ({
            base: current?.base ?? node,
            title: current?.title ?? node.title,
            config: next,
          }))
        }
      />
      <CanvasNodeInputs
        key={node.id}
        node={node}
        capability={capability}
        disabled={!editable}
        unsaved={!!draft}
      />
      {draft && draft.base.configVersion !== node.configVersion && (
        <CanvasNodeConflicts
          key={node.configVersion}
          draft={draft}
          current={node}
          onRebase={setDraft}
          disabled={pending}
        />
      )}
      {storageError && (
        <p role="alert" className="text-sm text-destructive">
          {storageError}
        </p>
      )}
      {invalid && (
        <Button variant="outline" size="sm" onClick={() => setDraft(null)}>
          清理无效草稿
        </Button>
      )}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </section>
  );
  return (
    <div>
      {resource.loading && !resource.data && (
        <p role="status" className="p-4 text-sm text-muted-foreground">
          正在读取项目内容…
        </p>
      )}
      {resource.error && (
        <p role="alert" className="p-4 text-sm text-destructive">
          {resource.error}
        </p>
      )}
      {resource.data && (
        <>
          <CanvasResourceMedia resource={resource.data} onPreview={onPreview} />
          <CanvasResourceEditor
            key={`${resource.data.header.resourceType}:${resource.data.header.resourceId}`}
            resource={resource.data}
            onPreview={onPreview}
          />
          <CanvasResourceActions
            key={`actions:${resource.data.header.resourceType}:${resource.data.header.resourceId}`}
            resource={resource.data}
          />
        </>
      )}
      {node.resource ? (
        <details
          className="border-t border-border/20"
          open={!!draft || invalid}
        >
          <summary className="cursor-pointer p-4 text-sm font-medium">
            画布显示与引用{draft ? " · 有未保存修改" : ""}
          </summary>
          {settings}
        </details>
      ) : (
        settings
      )}
      <Dialog open={rebinding} onOpenChange={setRebinding}>
        <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden">
          <DialogHeader className="shrink-0">
            <DialogTitle>更换内容引用</DialogTitle>
            <DialogDescription>
              更换当前节点引用，其他引用节点保持各自绑定。
            </DialogDescription>
          </DialogHeader>
          <DialogScrollBody>
            {capability.resourceType && (
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
            )}
          </DialogScrollBody>
        </DialogContent>
      </Dialog>
    </div>
  );
}
