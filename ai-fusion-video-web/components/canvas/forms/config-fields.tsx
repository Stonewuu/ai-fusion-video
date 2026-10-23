"use client";

import { Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { CanvasCapability, CanvasConfig } from "@/lib/canvas/types";
import {
  advancedConfigFields,
  configuredImageSize,
  configFieldOrder,
  configFieldChoices,
  configFieldError,
} from "@/lib/canvas/field-presentation";
import { agentFormFields, canvasAgentKinds } from "@/lib/canvas/agent-types";
import { useCanvas } from "../canvas-provider";
import { CanvasModelFields } from "./model-fields";
import { CanvasSelect } from "./canvas-select";
import { CanvasBatchFields } from "./batch-fields";
import { CanvasWritebackFields } from "./writeback-fields";
import { CanvasAgentFields } from "./agent-fields";
import { CanvasMcpFields } from "./mcp-fields";
import { CanvasConfigFieldEditor } from "./config-field";
import { CanvasDimensionsFields } from "./dimensions-fields";
import { videoInputState, videoRulesOf } from "@/lib/video-model-rules";
import { modelForVersion } from "@/lib/canvas/model-constraints";

export function CanvasConfigFields({
  capability,
  config,
  onChange,
  connectedPorts = [],
  inputCounts = {},
  disabled = false,
}: {
  capability: CanvasCapability;
  config: CanvasConfig;
  onChange: (next: CanvasConfig) => void;
  connectedPorts?: string[];
  inputCounts?: Record<string, number>;
  disabled?: boolean;
}) {
  const catalog = useCanvas((state) => state.catalog!);
  const selectedModel = catalog.models.find((item) => item.id === config.modelId);
  const model = selectedModel ? modelForVersion(selectedModel, config.workflowVersionId) : undefined;
  const videoInputs = videoInputState(config, connectedPorts);
  const version = model?.versions.find(
    (item) => item.id === config.workflowVersionId,
  );
  const update = (key: string, value: CanvasConfig[string] | undefined) => {
    const next = { ...config };
    if (value === undefined) {
      delete next[key];
    } else {
      next[key] = value;
    }
    onChange(next);
  };
  if (capability.kind === "mcpTool") {
    return (
      <CanvasMcpFields
        config={config}
        onChange={onChange}
        connectedPorts={connectedPorts}
        disabled={disabled}
      />
    );
  }
  if (capability.kind === "writeback") {
    return (
      <CanvasWritebackFields
        config={config}
        onChange={onChange}
        connectedPorts={connectedPorts}
        disabled={disabled}
      />
    );
  }
  if (capability.kind === "foreach") {
    return (
      <CanvasBatchFields
        capability={capability}
        config={config}
        onChange={onChange}
        connectedPorts={connectedPorts}
        disabled={disabled}
      />
    );
  }
  const isAgent = canvasAgentKinds.includes(capability.kind);
  const dimensions =
    !!capability.configFields.width &&
    !!capability.configFields.height &&
    (!version ||
      (version.fields.includes("width") && version.fields.includes("height")));
  const fields = Object.entries(capability.configFields)
    .filter(([key]) => !isAgent || !agentFormFields.has(key))
    .filter(
      ([key]) =>
        key !== "frameType" ||
        capability.kind !== "agent" ||
        String(config.agentType).startsWith("storyboard_frame"),
    )
    .filter(
      ([key, field]) =>
        ![
          "resourceId",
          "modelId",
          "workflowVersionId",
          "selectedOutputId",
          "apiConfigId",
        ].includes(key) &&
        !(capability.kind === "comfyui" && key === "resultType") &&
        !(field.enumValues.length === 1 && config[key] === field.enumValues[0]),
    )
    .filter(
      ([key]) =>
        !version || version.fields.includes(key) || config[key] !== undefined,
    )
    .filter(([key]) => {
      const flag = {
        seed: "supportSeed",
        watermark: "supportWatermark",
        cameraFixed: "supportCameraFixed",
        generateAudio: "supportGenerateAudio",
      }[key];
      return (
        !flag || model?.parameters[flag] !== false || config[key] !== undefined
      );
    })
    .filter(([key]) => !dimensions || !["width", "height"].includes(key))
    .filter(
      ([key]) =>
        capability.kind !== "imageTransform" ||
        (config.operation === "crop"
          ? key !== "fit"
          : !["x", "y"].includes(key)),
    )
    .sort(
      ([a], [b]) =>
        (configFieldOrder.includes(a) ? configFieldOrder.indexOf(a) : 100) -
        (configFieldOrder.includes(b) ? configFieldOrder.indexOf(b) : 100),
    );
  const primary = fields.filter(([key]) => !advancedConfigFields.has(key));
  const advanced = fields.filter(([key]) => advancedConfigFields.has(key));
  const advancedError = advanced.some(([key, field]) =>
    configFieldError(
      key,
      field,
      config[key],
      configFieldChoices(capability.kind, key, field, model),
      model,
    ),
  );
  const imageSize = configuredImageSize(config, model);
  const renderFields = (items: typeof fields) =>
    items.map(([key, field]) => (
      <CanvasConfigFieldEditor
        key={key}
        kind={capability.kind}
        name={key}
        field={field}
        config={config}
        model={model}
        connected={connectedPorts.includes(key)}
        videoInputs={videoInputs}
        disabled={disabled}
        onChange={update}
      />
    ));
  return (
    <div className="flex min-w-0 flex-col gap-5">
      {capability.configFields.modelId && (
        <CanvasModelFields
          capability={capability}
          config={config}
          onChange={onChange}
          connectedPorts={connectedPorts}
          inputCounts={inputCounts}
          disabled={disabled}
        />
      )}
      {capability.configFields.apiConfigId && (
        <div className="flex flex-col gap-2">
          <span className="text-sm font-medium">音频服务 *</span>
          <CanvasSelect
            label="音频服务"
            value={
              config.apiConfigId === undefined
                ? null
                : String(config.apiConfigId)
            }
            placeholder="选择音频服务"
            disabled={disabled}
            items={catalog.audioServices.map((service) => ({
              value: String(service.id),
              label: service.name,
            }))}
            onChange={(next) => update("apiConfigId", Number(next))}
          />
          {!catalog.audioServices.length && (
            <p className="text-xs text-muted-foreground">
              请先在系统设置中启用音频服务，再刷新画布。
            </p>
          )}
        </div>
      )}
      {capability.kind === "selection" && (
        <div className="flex flex-col gap-2">
          <p className="text-xs text-muted-foreground">
            运行后在候选结果中选用一项，后续步骤将继续执行。
          </p>
          {config.selectedOutputId && (
            <Button
              variant="outline"
              disabled={disabled}
              onClick={() => update("selectedOutputId", undefined)}
            >
              重新选择候选
            </Button>
          )}
        </div>
      )}
      {renderFields(primary)}
      {videoRulesOf(model?.parameters)?.fixedGenerateAudio === true && (
        <p className="text-xs text-muted-foreground">
          此模型会同时生成视频声音。
        </p>
      )}
      {dimensions && (
        <CanvasDimensionsFields
          capability={capability}
          config={config}
          onChange={onChange}
          disabled={disabled}
        />
      )}
      {imageSize && (
        <p className="text-xs text-primary">
          预计画面尺寸：{imageSize.replace("x", " × ")} px
          {config.width || config.height ? "（已另设自定义尺寸）" : ""}
        </p>
      )}
      {isAgent && (
        <CanvasAgentFields
          capability={capability}
          config={config}
          onChange={onChange}
          disabled={disabled}
          connectedPorts={connectedPorts}
        />
      )}
      {!!advanced.length && (
        <details className="min-w-0 border-t border-border/20 pt-4">
          <summary className="cursor-pointer rounded-lg text-sm font-medium outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">
            <Settings2 className="mr-2 inline size-4 text-muted-foreground" />
            高级设置
            {advancedError ? (
              <span className="ml-2 text-xs font-normal text-destructive">
                有设置需调整
              </span>
            ) : (
              advanced.some(([key]) => config[key] !== undefined) && (
                <span className="ml-2 text-xs font-normal text-primary">
                  已自定义
                </span>
              )
            )}
          </summary>
          <div className="flex min-w-0 flex-col gap-4 pt-4">
            {renderFields(advanced)}
          </div>
        </details>
      )}
    </div>
  );
}
