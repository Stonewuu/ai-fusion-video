"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { toast } from "sonner";
import {
  Check,
  CornerDownRight,
  Crop,
  Expand,
  Film,
  ImageIcon,
  Loader2,
  PenLine,
  Replace,
  Scissors,
  SlidersHorizontal,
  Video,
  X,
  type LucideIcon,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { CanvasInput } from "../ui/canvas-field";
import { CanvasPanel } from "../ui/canvas-surface";
import { CanvasNodeTypeIcon } from "../shared/canvas-node-icon";
import { CanvasMediaPicker } from "../library/media-picker";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { ManagedMedia } from "@/lib/api/media";
import { compatibleNodes } from "@/lib/canvas/graph";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import { canvasMediaKinds } from "@/lib/canvas/placement";
import { nodeMedia, type CanvasMediaPreview } from "@/lib/canvas/presentation";
import type { CanvasNode, CanvasNodeKind } from "@/lib/canvas/types";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import type { CanvasNodeChoice } from "../editor/node-menu";

interface MediaNextStep {
  kind: CanvasNodeKind;
  portId: string;
  label: string;
  hint: string;
  icon: LucideIcon;
}

const mediaNextSteps: Partial<Record<CanvasNodeKind, MediaNextStep[]>> = {
  image: [
    {
      kind: "imageGeneration",
      portId: "referenceImages",
      label: "生成图片",
      hint: "新建「生成图片」节点，作为参考图接入",
      icon: ImageIcon,
    },
    {
      kind: "videoGeneration",
      portId: "firstFrame",
      label: "生成视频",
      hint: "新建「生成视频」节点，作为首帧接入",
      icon: Video,
    },
    {
      kind: "imageTransform",
      portId: "image",
      label: "裁剪缩放",
      hint: "新建「裁剪与调整尺寸」节点并连线",
      icon: Crop,
    },
  ],
  video: [
    {
      kind: "frameExtract",
      portId: "video",
      label: "提取首尾帧",
      hint: "新建「提取首尾帧」节点并连线",
      icon: Scissors,
    },
    {
      kind: "videoGeneration",
      portId: "referenceVideos",
      label: "生成视频",
      hint: "新建「生成视频」节点，作为参考视频接入",
      icon: Video,
    },
    {
      kind: "videoCompose",
      portId: "videos",
      label: "合成视频",
      hint: "新建「合成视频」节点并连线",
      icon: Film,
    },
  ],
  audio: [
    {
      kind: "audioMix",
      portId: "music",
      label: "混音",
      hint: "新建「混音」节点并连线",
      icon: SlidersHorizontal,
    },
    {
      kind: "videoGeneration",
      portId: "referenceAudios",
      label: "生成视频",
      hint: "新建「生成视频」节点，作为参考音频接入",
      icon: Video,
    },
  ],
  textFile: [
    {
      kind: "imageGeneration",
      portId: "prompt",
      label: "生成图片",
      hint: "新建「生成图片」节点并连线",
      icon: ImageIcon,
    },
    {
      kind: "videoGeneration",
      portId: "prompt",
      label: "生成视频",
      hint: "新建「生成视频」节点并连线",
      icon: Video,
    },
    {
      kind: "textGeneration",
      portId: "prompt",
      label: "生成文本",
      hint: "新建「生成文本」节点并连线",
      icon: PenLine,
    },
  ],
};

/** 媒体素材节点（图片、视频、音频、文本文件、文档）的紧凑悬浮面板。 */
export function MediaNodePanel({
  node,
  maxHeight,
  onClose,
  onPreview,
  onCreateNext,
}: {
  node: CanvasNode;
  maxHeight: number;
  onClose: () => void;
  onPreview: (media: CanvasMediaPreview, returnFocus?: HTMLElement) => void;
  onCreateNext: (portId: string, choice: CanvasNodeChoice) => void;
}) {
  const store = useCanvasStoreApi();
  const snapshot = useCanvas((state) => state.snapshot!);
  const catalog = useCanvas((state) => state.catalog!);
  const pending = useCanvas((state) => !!state.pending);
  const capability = catalog.nodes.find((item) => item.kind === node.kind)!;
  const mediaKind =
    canvasMediaKinds[node.kind as keyof typeof canvasMediaKinds];
  const hasMedia = Boolean(node.config.resourceId);
  const preview = nodeMedia(node, null);
  const [syncedTitle, setSyncedTitle] = useState(node.title);
  const [title, setTitle] = useState(node.title);
  if (node.title !== syncedTitle) {
    setSyncedTitle(node.title);
    if (title === syncedTitle) {
      setTitle(node.title);
    }
  }
  const [picking, setPicking] = useState(!hasMedia);
  const [saving, setSaving] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const saveTask = useRef<Promise<boolean> | null>(null);
  const savedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    return () => {
      if (savedTimer.current) {
        clearTimeout(savedTimer.current);
      }
    };
  }, []);
  const editable =
    snapshot.permissions.canEdit && !snapshot.document.archived && !pending;
  const dirty = title !== node.title;
  const output = capability.outputs[0];
  const compatible = output
    ? compatibleNodes(node, output, catalog.nodes, snapshot.document.projectId)
    : [];
  const steps = (mediaNextSteps[node.kind] ?? []).flatMap((step) => {
    const options = compatible.filter(
      (item) => item.capability.kind === step.kind,
    );
    const connection =
      options.find((item) => item.port.id === step.portId) ?? options[0];
    return connection ? [{ ...step, connection }] : [];
  });
  const markSaved = () => {
    setJustSaved(true);
    if (savedTimer.current) {
      clearTimeout(savedTimer.current);
    }
    savedTimer.current = setTimeout(() => {
      setJustSaved(false);
    }, 1800);
  };
  const saveTitle = async (nextTitle: string = title): Promise<boolean> => {
    if (saveTask.current) {
      const ok = await saveTask.current;
      if (!ok) {
        return false;
      }
    }
    const latest = store
      .getState()
      .snapshot?.graph.nodes.find((item) => item.id === node.id);
    if (!latest) {
      return false;
    }
    const trimmed = nextTitle.trim();
    if (!trimmed) {
      setError("节点名称不能为空");
      return false;
    }
    if (trimmed === latest.title) {
      setTitle(latest.title);
      setError(null);
      return true;
    }
    setSaving(true);
    setError(null);
    const task = (async () => {
      try {
        await store.getState().edit("重命名节点", [
          {
            type: "canvas.node.patch",
            nodeId: node.id,
            expectedConfigVersion: latest.configVersion,
            title: trimmed,
          },
        ]);
        markSaved();
        return true;
      } catch (failure) {
        const message = getApiErrorMessage(failure);
        setError(message);
        toast.error(message);
        return false;
      } finally {
        saveTask.current = null;
        setSaving(false);
      }
    })();
    saveTask.current = task;
    return task;
  };
  const pickMedia = async (media: ManagedMedia) => {
    if (saving || !editable) {
      return;
    }
    const titleSaved = await saveTitle();
    if (!titleSaved) {
      return;
    }
    const latest = store
      .getState()
      .snapshot?.graph.nodes.find((item) => item.id === node.id);
    if (!latest) {
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await store.getState().edit("更换节点素材", [
        {
          type: "canvas.node.patch",
          nodeId: node.id,
          expectedConfigVersion: latest.configVersion,
          configPatch: { resourceId: media.id },
        },
      ]);
      setPicking(false);
      markSaved();
    } catch (failure) {
      const message = getApiErrorMessage(failure);
      setError(message);
      toast.error(message);
    } finally {
      setSaving(false);
    }
  };
  const close = async () => {
    if (await saveTitle()) {
      onClose();
    }
  };
  const showBody = picking || !hasMedia || Boolean(error);
  return (
    <CanvasPanel
      aria-label={`编辑${capability.label}`}
      data-canvas-generation-composer
      style={
        {
          maxHeight,
          "--canvas-tone": canvasNodeColor(node.kind),
        } as CSSProperties
      }
      onKeyDown={(event) => {
        if (event.key === "Escape" && !event.nativeEvent.isComposing) {
          event.preventDefault();
          void close();
        }
      }}
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
            aria-label="节点名称"
            placeholder="节点名称"
            value={title}
            maxLength={200}
            disabled={!editable}
            onChange={(event) => {
              setJustSaved(false);
              setError(null);
              setTitle(event.target.value);
            }}
            onBlur={() => {
              if (dirty) {
                void saveTitle();
              }
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.nativeEvent.isComposing) {
                event.preventDefault();
                void saveTitle();
              }
            }}
          />
        </div>
        {preview && (
          <CanvasIconButton
            label="放大预览"
            onClick={(event) => onPreview(preview, event.currentTarget)}
          >
            <Expand />
          </CanvasIconButton>
        )}
        <CanvasButton
          variant="secondary"
          aria-pressed={picking}
          disabled={!editable || saving}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => setPicking((current) => !current)}
        >
          <Replace className="size-3.5 text-muted-foreground" />
          {hasMedia ? (picking ? "收起素材" : "更换媒体") : "选择媒体"}
        </CanvasButton>
        <CanvasIconButton
          label="收起"
          shortcut="Esc"
          disabled={saving}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => void close()}
        >
          <X />
        </CanvasIconButton>
      </div>
      {showBody && (
        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto border-t border-foreground/[0.07] px-3 py-2.5 dark:border-white/[0.08]">
          <div className="flex min-w-0 flex-col gap-3 p-1">
            {error && (
              <p role="alert" className="text-xs text-destructive">
                {error}
              </p>
            )}
            {(picking || !hasMedia) && mediaKind && (
              <CanvasMediaPicker
                kind={mediaKind}
                surface="canvas"
                disabled={!editable || saving}
                value={String(node.config.resourceId ?? "")}
                onSelect={(media) => void pickMedia(media)}
              />
            )}
          </div>
        </div>
      )}
      <footer className="flex shrink-0 flex-wrap items-center gap-1.5 border-t border-foreground/[0.07] bg-foreground/[0.02] px-3 py-2.5 dark:border-white/[0.08] dark:bg-white/[0.025]">
        {steps.length > 0 ? (
          <div
            role="group"
            aria-label="下一步新建节点"
            className="flex min-w-0 flex-wrap items-center gap-1"
          >
            <span className="inline-flex shrink-0 items-center gap-1 pr-0.5 text-xs text-muted-foreground">
              <CornerDownRight
                className="size-3.5 text-muted-foreground/75"
                aria-hidden
              />
              <span>下一步</span>
            </span>
            {steps.map((step) => {
              const stepDisabled = !editable || saving || !hasMedia;
              return (
                <Tooltip key={step.kind}>
                  <TooltipTrigger
                    render={
                      <CanvasButton
                        variant="ghost"
                        disabled={stepDisabled}
                        onMouseDown={(event) => event.preventDefault()}
                        onClick={async () => {
                          if (await saveTitle()) {
                            onCreateNext(output!.id, {
                              capability: step.connection.capability,
                              connection: step.connection,
                            });
                          }
                        }}
                      />
                    }
                  >
                    <step.icon
                      aria-hidden
                      style={{ color: canvasNodeColor(step.kind) }}
                    />
                    <span>{step.label}</span>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">{step.hint}</TooltipContent>
                </Tooltip>
              );
            })}
          </div>
        ) : (
          <span role="status" className="pl-0.5 text-xs text-muted-foreground">
            {justSaved ? "已保存" : `${capability.label} · 可连入智能体使用`}
          </span>
        )}
        <CanvasButton
          variant={dirty ? "generate" : "secondary"}
          className="ml-auto min-w-20"
          disabled={dirty ? !editable || saving : saving}
          title={dirty ? "保存名称" : "收起面板（Esc）"}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => {
            if (dirty) {
              void saveTitle();
            } else {
              void close();
            }
          }}
        >
          {saving ? (
            <Loader2 className="animate-spin motion-reduce:animate-none" />
          ) : dirty ? (
            <Check className="size-3.5" />
          ) : null}
          {saving ? "保存中" : dirty ? "保存" : "完成"}
        </CanvasButton>
      </footer>
    </CanvasPanel>
  );
}
