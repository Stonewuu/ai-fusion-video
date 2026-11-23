"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { toast } from "sonner";
import {
  Check,
  CornerDownRight,
  ImageIcon,
  Loader2,
  PenLine,
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
import { CanvasTextarea } from "../ui/canvas-field";
import { CanvasPanel } from "../ui/canvas-surface";
import { CanvasNodeTypeIcon } from "../shared/canvas-node-icon";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { compatibleNodes } from "@/lib/canvas/graph";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import type { CanvasNode, CanvasNodeKind } from "@/lib/canvas/types";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import type { CanvasNodeChoice } from "../editor/node-menu";

/** 快捷新建的下游节点：图标与颜色与画布节点类型保持一致。 */
const nextSteps: {
  kind: CanvasNodeKind;
  label: string;
  hint: string;
  icon: LucideIcon;
}[] = [
  {
    kind: "imageGeneration",
    label: "生成图片",
    hint: "新建「生成图片」节点并连线",
    icon: ImageIcon,
  },
  {
    kind: "videoGeneration",
    label: "生成视频",
    hint: "新建「生成视频」节点并连线",
    icon: Video,
  },
  {
    kind: "textGeneration",
    label: "生成文本",
    hint: "新建「生成文本」节点并连线，结果可写回正文",
    icon: PenLine,
  },
];

/** 文本节点的快捷面板：直接编写正文，支持即时保存与一键新建已连线的下游生成节点。 */
export function TextComposer({
  node,
  maxHeight,
  onClose,
  onCreateNext,
}: {
  node: CanvasNode;
  maxHeight: number;
  onClose: () => void;
  onCreateNext: (portId: string, choice: CanvasNodeChoice) => void;
}) {
  const store = useCanvasStoreApi();
  const snapshot = useCanvas((state) => state.snapshot!);
  const catalog = useCanvas((state) => state.catalog!);
  const pending = useCanvas((state) => !!state.pending);
  const capability = catalog.nodes.find((item) => item.kind === node.kind)!;
  const field = capability.configFields.content;
  const saved = String(node.config.content ?? "");
  const [syncedSaved, setSyncedSaved] = useState(saved);
  const [value, setValue] = useState(saved);
  if (saved !== syncedSaved) {
    setSyncedSaved(saved);
    if (value === syncedSaved) {
      setValue(saved);
    }
  }
  const [saving, setSaving] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
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
  const dirty = value !== saved;
  const output = capability.outputs[0];
  const compatible = output
    ? compatibleNodes(node, output, catalog.nodes, snapshot.document.projectId)
    : [];
  const steps = nextSteps.flatMap((step) => {
    const options = compatible.filter(
      (item) => item.capability.kind === step.kind,
    );
    const connection =
      options.find((item) => item.port.id === "prompt") ?? options[0];
    return connection ? [{ ...step, connection }] : [];
  });
  const save = async (content: string = value): Promise<boolean> => {
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
    if (content === String(latest.config.content ?? "")) {
      return true;
    }
    setSaving(true);
    const task = (async () => {
      try {
        await store.getState().edit("修改文本", [
          {
            type: "canvas.node.patch",
            nodeId: node.id,
            expectedConfigVersion: latest.configVersion,
            configPatch: { content },
          },
        ]);
        setJustSaved(true);
        if (savedTimer.current) {
          clearTimeout(savedTimer.current);
        }
        savedTimer.current = setTimeout(() => {
          setJustSaved(false);
        }, 1800);
        return true;
      } catch (failure) {
        toast.error(getApiErrorMessage(failure));
        return false;
      } finally {
        saveTask.current = null;
        setSaving(false);
      }
    })();
    saveTask.current = task;
    return task;
  };
  const close = async () => {
    if (await save()) {
      onClose();
    }
  };
  const heading = node.kind === "text" ? "正文" : capability.label;
  const placeholder =
    node.kind === "note"
      ? "记录想法、分工说明或评审备注…"
      : node.kind === "story"
        ? "写下故事大纲、情节设定或分场梗概…"
        : "写下故事、场景或角色设定，例如：雨后的小巷，一只橘猫坐在窗边。";
  return (
    <CanvasPanel
      aria-label={`编辑${heading}`}
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
      <div className="flex shrink-0 items-center gap-2 px-3 pb-1.5 pt-2.5">
        <span className="canvas-node-badge-icon shrink-0">
          <CanvasNodeTypeIcon
            capability={capability}
            className="size-3.5"
            aria-hidden
          />
        </span>
        <span className="text-[13px] font-medium">{heading}</span>
        <span
          role="status"
          className="flex min-w-0 flex-1 items-center gap-1.5 text-xs text-muted-foreground"
        >
          {saving ? (
            "正在保存…"
          ) : dirty ? (
            <>
              <span className="size-1.5 rounded-full bg-amber-500" aria-hidden />
              未保存
            </>
          ) : justSaved ? (
            "已保存"
          ) : null}
        </span>
        {field?.maxLength && (
          <span className="text-[11px] tabular-nums text-muted-foreground/80">
            {[...value].length.toLocaleString()} /{" "}
            {field.maxLength.toLocaleString()}
          </span>
        )}
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
      <div className="flex min-h-0 flex-col overflow-y-auto px-3 pb-3 pt-1">
        <div className="py-0.5">
          <label htmlFor={`text-content-${node.id}`} className="sr-only">
            {heading}
          </label>
          <CanvasTextarea
            id={`text-content-${node.id}`}
            autoFocus={!saved}
            rows={5}
            disabled={!editable}
            className="h-32"
            placeholder={placeholder}
            value={value}
            maxLength={field?.maxLength ?? undefined}
            onChange={(event) => {
              setJustSaved(false);
              setValue(event.target.value);
            }}
            onBlur={() => void save()}
            onKeyDown={(event) => {
              if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.preventDefault();
                void save();
              }
            }}
          />
        </div>
      </div>
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
              const stepDisabled = !editable || saving || !value.trim();
              return (
                <Tooltip key={step.kind}>
                  <TooltipTrigger
                    render={
                      <CanvasButton
                        variant="ghost"
                        disabled={stepDisabled}
                        onMouseDown={(event) => event.preventDefault()}
                        onClick={async () => {
                          if (await save()) {
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
          <span className="pl-0.5 text-xs text-muted-foreground">
            画布备注 · 仅用于说明，不参与运行
          </span>
        )}
        <CanvasButton
          variant={dirty ? "generate" : "secondary"}
          className="ml-auto min-w-20"
          disabled={dirty ? !editable || saving : saving}
          title={dirty ? "保存修改（Ctrl/⌘ Enter）" : "收起正文面板（Esc）"}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => {
            if (dirty) {
              void save();
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
