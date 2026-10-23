"use client";
import { CanvasGuidance } from "../shared/canvas-guidance";

import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getApiErrorMessage } from "@/lib/api/api-error";
import {
  modelFitsInputs,
  modelForVersion,
  modelInputs,
  modelKind,
  modelsForNode,
} from "@/lib/canvas/model-constraints";
import { canvasPortLabels, canvasTypeNames } from "@/lib/canvas/labels";
import { canvasAgentKinds } from "@/lib/canvas/agent-types";
import type {
  CanvasCapability,
  CanvasConfig,
  CanvasModelOption,
  CanvasValueType,
} from "@/lib/canvas/types";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasSelect } from "./canvas-select";

export function CanvasModelFields({
  capability,
  config,
  onChange,
  connectedPorts,
  inputCounts,
  disabled,
}: {
  capability: CanvasCapability;
  config: CanvasConfig;
  onChange: (value: CanvasConfig) => void;
  connectedPorts: string[];
  inputCounts: Record<string, number>;
  disabled: boolean;
}) {
  const store = useCanvasStoreApi();
  const catalog = useCanvas((state) => state.catalog!);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const literalPorts = capability.inputs
    .filter((port) => Object.hasOwn(config, port.id))
    .map((port) => port.id);
  const ports = [...new Set([...connectedPorts, ...literalPorts])];
  const counts = {
    ...inputCounts,
    ...Object.fromEntries(literalPorts.map((port) => [port, 1])),
  };
  const availableModels = modelsForNode(capability.kind, catalog);
  const matchesInputs = (model: CanvasModelOption) => {
    if (!modelKind(capability.kind)) return true;
    return model.kind === "comfyui"
      ? model.versions.some((version) =>
          !version.unavailable && modelFitsInputs(modelForVersion(model, version.id), version.inputs, ports, counts),
        )
      : modelFitsInputs(model, model.inputs, ports, counts);
  };
  const options = [...availableModels].sort(
    (a, b) => Number(matchesInputs(b)) - Number(matchesInputs(a)),
  );
  const selectedModel = catalog.models.find((item) => item.id === config.modelId);
  const model = selectedModel ? modelForVersion(selectedModel, config.workflowVersionId) : undefined;
  const modelLabel = canvasAgentKinds.includes(capability.kind) ? "思考模型" : "模型";
  const inputs = model ? modelInputs(model, config) : undefined;
  const refresh = async () => {
    setRefreshing(true);
    try {
      await store.getState().refreshCatalog();
      setError(null);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setRefreshing(false);
    }
  };
  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium">
          {modelLabel}{modelKind(capability.kind) || canvasAgentKinds.includes(capability.kind) ? " *" : ""}
        </span>
        <Button
          variant="ghost"
          size="icon-xs"
          title="刷新模型列表"
          disabled={refreshing}
          onClick={() => void refresh()}
        >
          <RefreshCw />
        </Button>
      </div>
      <CanvasSelect
        label={modelLabel}
        value={
          options.some((item) => item.id === config.modelId)
            ? String(config.modelId)
            : null
        }
        items={options.map((item) => ({
          value: String(item.id),
          label: `${item.name}${matchesInputs(item) ? "" : " · 输入需调整"}`,
        }))}
        disabled={disabled || refreshing}
        placeholder={options.length ? "选择模型" : "暂无可用模型"}
        onChange={(value) => {
          const next: CanvasConfig = { ...config, modelId: Number(value) };
          delete next.workflowVersionId;
          if (capability.kind === "comfyui")
            next.resultType = options.find(
              (item) => item.id === Number(value),
            )!.resultType;
          onChange(next);
        }}
      />
      {model && !matchesInputs(model) && (
        <p role="alert" className="text-xs text-destructive">
          当前模型不支持这些输入。可以继续编辑，运行前请调整连接或更换模型。
        </p>
      )}
      {!options.length && (
        <CanvasGuidance title="先添加一个可用模型" tone="warning">
          在系统设置中添加并启用对应模型，然后刷新列表。没有配置权限时，请联系项目管理员。
          <Button variant="outline" size="sm" disabled={refreshing} onClick={() => void refresh()}><RefreshCw />刷新模型</Button>
        </CanvasGuidance>
      )}
      {capability.kind === "comfyui" && model && (
        <div className="flex flex-col gap-2 pt-2">
          <p className="text-sm font-medium">工作流版本</p>
          <CanvasSelect
            label="工作流版本"
            placeholder="选择已发布版本"
            disabled={disabled}
            value={
              config.workflowVersionId === undefined
                ? null
                : String(config.workflowVersionId)
            }
            items={model.versions.map((version) => ({
                value: String(version.id),
                label: `版本 ${version.number}${version.unavailable ? " · 配置冲突" : modelFitsInputs(modelForVersion(selectedModel!, version.id), version.inputs, ports, counts) ? "" : " · 输入需调整"}`,
              }))}
            onChange={(value) =>
              onChange({ ...config, workflowVersionId: Number(value) })
            }
          />
          <p className="text-xs text-muted-foreground">
            输出：{canvasTypeNames[model.resultType as CanvasValueType]}
            。节点固定此版本，后续发布不会改变它。
          </p>
        </div>
      )}
      {inputs && (
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer rounded-lg outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">模型支持的输入</summary>
          <div className="flex flex-col gap-2 pt-2">
          <p>
            支持输入：
            {Object.entries(inputs)
              .filter(([, rule]) => rule.maximum !== 0)
              .map(
                ([port, rule]) =>
                  `${canvasPortLabels[port] ?? port}${port !== "context" && rule.maximum !== null && rule.maximum > 1 ? `（最多 ${rule.maximum} 项）` : ""}`,
              )
              .join("、") || "无外部输入"}
          </p>
          {model?.parameters.exclusiveInputModes === true && (
            <p>首尾帧与多模态参考需选择一种方式。</p>
          )}
          {model?.parameters.audioRequiresVisualInput === true && (
            <p>参考音频需要同时提供图片或视频。</p>
          )}
          </div>
        </details>
      )}
      {(error || model?.unavailable) && (
        <p role="alert" className="text-xs text-destructive">
          {error ?? model?.unavailable}
        </p>
      )}
    </section>
  );
}
