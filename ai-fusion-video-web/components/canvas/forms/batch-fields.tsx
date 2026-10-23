"use client";

import { useId } from "react";
import { Layers3 } from "lucide-react";
import { CanvasNumberField } from "./field-controls";
import { initialConfig, portType, portTypes } from "@/lib/canvas/graph";
import { canvasPortLabels, canvasTypeNames } from "@/lib/canvas/labels";
import { canvasAgentKinds } from "@/lib/canvas/agent-types";
import type {
  CanvasCapability,
  CanvasConfig,
  CanvasPort,
  CanvasValueType,
} from "@/lib/canvas/types";
import { useCanvas } from "../canvas-provider";
import { CanvasSelect } from "./canvas-select";
import { CanvasConfigFields } from "./config-fields";

export function CanvasBatchFields({
  capability,
  config,
  onChange,
  connectedPorts,
  disabled,
}: {
  capability: CanvasCapability;
  config: CanvasConfig;
  onChange: (config: CanvasConfig) => void;
  connectedPorts: string[];
  disabled: boolean;
}) {
  const id = useId();
  const catalog = useCanvas((state) => state.catalog!);
  const allowed = capability.configFields.operationKind.enumValues;
  const operations = catalog.nodes.filter((node) =>
    allowed.includes(node.kind),
  );
  const operation = operations.find(
    (node) => node.kind === config.operationKind,
  );
  const body =
    typeof config.operationConfig === "object" && !Array.isArray(config.operationConfig) ? config.operationConfig : {};
  const lockedType = connectedPorts.includes("items")
    ? String(config.inputType)
    : undefined;
  const defaults = (node: CanvasCapability): CanvasConfig => ({
    ...initialConfig(node),
    ...(node.kind === "comfyui" ? { resultType: "image" } : {}),
    ...(node.kind === "agent"
      ? { inputType: lockedType ?? "text", resultType: "text" }
      : {}),
  });
  const inputOptions = (node: CanvasCapability, next: CanvasConfig) =>
    node.inputs.filter((port) => {
      if (
        lockedType &&
        !portTypes(port, next).includes(lockedType as CanvasValueType)
      )
        return false;
      return node.inputs.every(
        (other) =>
          other.id === port.id ||
          !other.required ||
          portType(other, next) === "text",
      );
    });
  const outputOptions = (node: CanvasCapability, next: CanvasConfig) =>
    node.outputs.filter(
      (port) =>
        node.kind !== "frameExtract" ||
        next.frameType === "both" ||
        next.frameType === undefined ||
        port.id === `${next.frameType}Frame`,
    );
  const change = (
    node: CanvasCapability,
    next: CanvasConfig,
    inputId?: string,
    outputId?: string,
  ) => {
    const input =
      inputOptions(node, next).find((port) => port.id === inputId) ??
      inputOptions(node, next)[0];
    const output =
      outputOptions(node, next).find((port) => port.id === outputId) ??
      outputOptions(node, next)[0];
    const updated = { ...next };
    if (input) delete updated[input.id];
    const types = input ? portTypes(input, updated) : [];
    const selectedType = (lockedType ?? config.inputType) as CanvasValueType;
    onChange({
      ...config,
      operationKind: node.kind,
      operationConfig: updated,
      inputPort: input?.id ?? "",
      outputPort: output?.id ?? "",
      inputType: types.includes(selectedType) ? selectedType : (types[0] ?? ""),
      resultType: output ? portType(output, updated)! : "",
    });
  };
  const label = (port: CanvasPort) =>
    `${canvasPortLabels[port.id] ?? port.label}（${portTypes(port, body).length > 1 ? "多种内容" : (canvasTypeNames[portType(port, body)!] ?? "未设置类型")}）`;
  const selectedInput = operation?.inputs.find(
    (port) => port.id === config.inputPort,
  );
  const inputTypes = selectedInput ? portTypes(selectedInput, body) : [];
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium">对每一项执行 *</label>
        <CanvasSelect
          label="对每一项执行"
          value={
            typeof config.operationKind === "string"
              ? config.operationKind
              : null
          }
          placeholder="选择处理方式"
          disabled={disabled}
          items={operations
            .filter((node) => inputOptions(node, defaults(node)).length > 0)
            .map((node) => ({ value: node.kind, label: node.label }))}
          onChange={(value) => {
            const node = operations.find((item) => item.kind === value)!;
            change(node, defaults(node));
          }}
        />
      </div>
      {operation && (
        <>
          {canvasAgentKinds.includes(operation.kind) ? (
            <p className="text-sm text-muted-foreground">
              专业智能体按原顺序逐项执行，读取、编辑和生成自动批准。每项保留独立任务记录，运行预览会显示总用量范围。
            </p>
          ) : null}
          <section className="flex flex-col gap-3 rounded-lg border border-border/20 bg-background/70 p-3">
            <p className="flex items-center gap-2 text-sm font-medium">
              <Layers3 className="size-4 text-primary" />
              逐项处理，按原顺序汇总
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="flex flex-col gap-2">
                <label className="text-sm font-medium">每项作为</label>
                <CanvasSelect
                  label="每项作为"
                  placeholder="选择输入"
                  value={String(config.inputPort ?? "")}
                  disabled={disabled}
                  items={inputOptions(operation, body).map((port) => ({
                    value: port.id,
                    label: label(port),
                  }))}
                  onChange={(value) =>
                    change(operation, body, value, String(config.outputPort))
                  }
                />
              </div>
              <div className="flex flex-col gap-2">
                <label className="text-sm font-medium">汇总结果</label>
                <CanvasSelect
                  label="汇总结果"
                  placeholder="选择输出"
                  value={String(config.outputPort ?? "")}
                  disabled={disabled}
                  items={outputOptions(operation, body).map((port) => ({
                    value: port.id,
                    label: label(port),
                  }))}
                  onChange={(value) =>
                    change(operation, body, String(config.inputPort), value)
                  }
                />
              </div>
            </div>
          </section>
          {inputTypes.length > 1 && !lockedType && (
            <CanvasSelect
              label="逐项素材类型"
              placeholder="选择素材类型"
              value={String(config.inputType ?? "")}
              disabled={disabled}
              items={inputTypes.map((type) => ({
                value: type,
                label: canvasTypeNames[type],
              }))}
              onChange={(value) => onChange({ ...config, inputType: value })}
            />
          )}
          <CanvasConfigFields
            capability={operation}
            config={body}
            connectedPorts={[String(config.inputPort)]}
            inputCounts={{ [String(config.inputPort)]: 1 }}
            disabled={disabled}
            onChange={(next) =>
              change(
                operation,
                next,
                String(config.inputPort),
                String(config.outputPort),
              )
            }
          />
        </>
      )}
      <div className="flex flex-col gap-2">
        <label htmlFor={`${id}-limit`} className="text-sm font-medium">
          最多处理多少项 *
        </label>
        <CanvasNumberField
          id={`${id}-limit`}
          label="最多处理多少项"
          min={1}
          max={1000}
          value={typeof config.maxItems === "number" ? config.maxItems : undefined}
          presets={[10, 25, 50, 100]}
          unit="项"
          disabled={disabled}
          onChange={(value) => {
            const next = { ...config };
            if (value === undefined) delete next.maxItems;
            else next.maxItems = value;
            onChange(next);
          }}
        />
        <p className="text-xs text-muted-foreground">
          运行前按此上限核算数量；只处理实际收到的项目，超出上限会停止。
        </p>
      </div>
    </div>
  );
}
