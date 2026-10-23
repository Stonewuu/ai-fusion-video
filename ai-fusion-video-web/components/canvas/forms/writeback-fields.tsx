"use client";

import { FileCheck2 } from "lucide-react";
import { canvasTypeNames } from "@/lib/canvas/labels";
import type { CanvasConfig, CanvasWritebackField } from "@/lib/canvas/types";
import { useCanvas } from "../canvas-provider";
import { CanvasSelect } from "./canvas-select";

export function CanvasWritebackFields({
  config,
  onChange,
  connectedPorts,
  disabled,
}: {
  config: CanvasConfig;
  onChange: (value: CanvasConfig) => void;
  connectedPorts: string[];
  disabled: boolean;
}) {
  const fields = useCanvas((state) => state.catalog!.writebackFields);
  const available = fields.filter(
    (field) =>
      (!connectedPorts.includes("value") ||
        field.valueType === config.valueType) &&
      (!connectedPorts.includes("target") ||
        field.targetType === config.targetType),
  );
  const targets = [...new Set(available.map((field) => field.targetType))];
  const bindings = available.filter(
    (field) => field.targetType === config.targetType,
  );
  const selected = fields.find((field) => field.key === config.binding);
  const select = (field: CanvasWritebackField) =>
    onChange({
      binding: field.key,
      targetType: field.targetType,
      valueType: field.valueType,
    });
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium">写入哪类内容 *</label>
        <CanvasSelect
          label="写入哪类内容"
          placeholder="选择目标类型"
          disabled={disabled || connectedPorts.includes("target")}
          value={
            typeof config.targetType === "string" ? config.targetType : null
          }
          items={targets.map((type) => ({
            value: type,
            label: canvasTypeNames[type],
          }))}
          onChange={(value) =>
            select(available.find((field) => field.targetType === value)!)
          }
        />
      </div>
      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium">写入哪个字段 *</label>
        <CanvasSelect
          label="写入哪个字段"
          placeholder="选择要更新的字段"
          disabled={disabled || !bindings.length}
          value={typeof config.binding === "string" ? config.binding : null}
          items={bindings.map((field) => ({
            value: field.key,
            label: field.label,
          }))}
          onChange={(value) =>
            select(bindings.find((field) => field.key === value)!)
          }
        />
      </div>
      {selected && (
        <p className="text-sm text-muted-foreground">
          将{canvasTypeNames[selected.valueType]}写入
          {canvasTypeNames[selected.targetType]}的“{selected.label}”。
        </p>
      )}
      <p className="flex items-start gap-2 text-xs leading-relaxed text-muted-foreground">
        <FileCheck2 className="mt-0.5 size-4 shrink-0 text-primary" />
        连接要采用的结果和目标内容节点。运行时先预览改动，确认后才写入；目标内容发生变化时会停止覆盖。
      </p>
    </div>
  );
}
