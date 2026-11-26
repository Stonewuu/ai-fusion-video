"use client";

import { useState, type ReactNode } from "react";
import { Ratio, SlidersHorizontal } from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { Dialog, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import {
  Popover,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import {
  AspectRatioPicker,
  ResolutionPicker,
} from "@/components/generation/media-output-pickers";
import { canvasImageSelection } from "@/lib/canvas/image-output";
import { configFieldChoices } from "@/lib/canvas/field-presentation";
import { modelForVersion } from "@/lib/canvas/model-constraints";
import { videoInputState } from "@/lib/video-model-rules";
import type { CanvasCapability, CanvasConfig } from "@/lib/canvas/types";
import { useCanvas } from "../canvas-provider";
import { CanvasDialogContent } from "../shared/canvas-dialog";
import { CanvasModelFields } from "../forms/model-fields";
import {
  CanvasConfigFields,
  canvasConfigFieldEntries,
} from "../forms/config-fields";
import { CanvasConfigFieldEditor } from "../forms/config-field";
import { CanvasImageOutputFields } from "../forms/image-output-fields";

/** 快捷生成底栏已直接编辑的参数：模型选择与输出设置弹层。 */
const composerFields: readonly string[] = [
  "modelId",
  "count",
  "ratio",
  "resolution",
  "duration",
];

export function GenerationParameters({
  capability,
  config,
  connectedPorts,
  inputCounts,
  edited,
  disabled,
  onChange,
  children,
}: {
  capability: CanvasCapability;
  config: CanvasConfig;
  connectedPorts: string[];
  inputCounts: Record<string, number>;
  /** 表单正文框正在编辑的字段，不在“更多参数”中重复出现。 */
  edited: string;
  disabled: boolean;
  onChange: (config: CanvasConfig) => void;
  children: ReactNode;
}) {
  const catalog = useCanvas((state) => state.catalog!);
  const [advanced, setAdvanced] = useState(false);
  const selected = catalog.models.find((item) => item.id === config.modelId);
  const model = selected
    ? modelForVersion(selected, config.workflowVersionId)
    : undefined;
  const image = capability.kind === "imageGeneration";
  // 输出设置弹层只服务生图与生视频；生成文本没有比例、分辨率、时长和数量。
  const video = capability.kind === "videoGeneration";
  const hasOutput = image || video;
  const omitted = new Set([...composerFields, edited]);
  const selection = model && image ? canvasImageSelection(config, model) : null;
  const videoInputs = videoInputState(config, connectedPorts);
  const remaining = canvasConfigFieldEntries(capability, config, model, {
    compact: true,
    omit: omitted,
  });
  const hasMore =
    remaining.fields.length > 0 || (!image && remaining.dimensions);
  const update = (key: string, value: CanvasConfig[string] | undefined) => {
    const next = { ...config };
    if (value === undefined) {
      delete next[key];
    } else {
      next[key] = value;
    }
    onChange(next);
  };
  const ratioChoices = video
    ? configFieldChoices(
      capability.kind,
      "ratio",
      capability.configFields.ratio,
      model,
      videoInputs,
    )
    : [];
  const resolutionChoices = video
    ? configFieldChoices(
      capability.kind,
      "resolution",
      capability.configFields.resolution,
      model,
    )
    : [];
  const summaryParts = [
    selection?.ratio || config.ratio,
    selection?.resolution || config.resolution,
    video &&
    (config.duration === -1
      ? "自动"
      : config.duration
        ? `${config.duration}s`
        : null),
    `×${config.count ?? 1}`,
  ].filter(Boolean);
  const summary = summaryParts.join(" · ");
  return (
    <>
      <div className="min-w-0 basis-full @[480px]:flex-1 @[480px]:basis-0">
        <CanvasModelFields
          capability={capability}
          config={config}
          connectedPorts={connectedPorts}
          inputCounts={inputCounts}
          disabled={disabled}
          compact
          onChange={onChange}
        />
      </div>
      {hasOutput && (
        <Popover>
          <PopoverTrigger
            render={
              <CanvasButton
                variant="secondary"
                data-field-trigger=""
                className="min-w-0 flex-1 px-2.5 text-xs tabular-nums @[480px]:flex-none"
                aria-label="输出参数"
                title={`输出设置：${summary}`}
                disabled={disabled}
              />
            }
          >
            <Ratio className="size-3.5 text-muted-foreground" />
            <span className="truncate">{summary}</span>
          </PopoverTrigger>
          <PopoverContent
            side="top"
            align="end"
            sideOffset={8}
            className="w-96 max-w-[calc(100vw-2rem)] rounded-[16px] border border-foreground/[0.1] shadow-xl dark:border-white/[0.12]"
          >
            <PopoverTitle>输出设置</PopoverTitle>
            <DialogScrollBody className="flex max-h-96 flex-col gap-5">
              {image ? (
                <CanvasImageOutputFields
                  capability={capability}
                  config={config}
                  model={model}
                  disabled={disabled}
                  onChange={onChange}
                />
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <AspectRatioPicker
                      value={String(config.ratio ?? "")}
                      options={ratioChoices.map((choice) => choice.value)}
                      disabled={disabled}
                      onChange={(value) => update("ratio", value)}
                    />
                    <ResolutionPicker
                      value={String(config.resolution ?? "")}
                      options={resolutionChoices}
                      disabled={disabled}
                      onChange={(value) => update("resolution", value)}
                    />
                  </div>
                  <CanvasConfigFieldEditor
                    kind={capability.kind}
                    name="duration"
                    field={capability.configFields.duration}
                    config={config}
                    model={model}
                    connected={false}
                    videoInputs={videoInputs}
                    disabled={disabled}
                    onChange={update}
                  />
                </>
              )}
              <CanvasConfigFieldEditor
                kind={capability.kind}
                name="count"
                field={capability.configFields.count}
                config={config}
                model={model}
                connected={false}
                disabled={disabled}
                onChange={update}
              />
            </DialogScrollBody>
          </PopoverContent>
        </Popover>
      )}
      {hasMore && (
        <CanvasIconButton
          variant="secondary"
          label="更多参数"
          onClick={() => setAdvanced(true)}
        >
          <SlidersHorizontal className="text-muted-foreground" />
        </CanvasIconButton>
      )}
      {children}
      <Dialog open={advanced && hasMore} onOpenChange={setAdvanced}>
        <CanvasDialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-xl">
          <DialogHeader className="shrink-0">
            <DialogTitle>更多参数</DialogTitle>
          </DialogHeader>
          <DialogScrollBody>
            <CanvasConfigFields
              capability={capability}
              config={config}
              compact
              omit={omitted}
              disabled={disabled}
              connectedPorts={connectedPorts}
              inputCounts={inputCounts}
              onChange={onChange}
            />
          </DialogScrollBody>
        </CanvasDialogContent>
      </Dialog>
    </>
  );
}
