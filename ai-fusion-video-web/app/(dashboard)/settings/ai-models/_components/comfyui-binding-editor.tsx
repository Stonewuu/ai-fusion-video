"use client";

import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type {
  ComfyUiBindingValueType,
  ComfyUiInputBindings,
  ComfyUiOutputBinding,
  ComfyUiOutputMediaType,
  ComfyUiOutputRole,
  ComfyUiWorkflowModelType,
} from "@/lib/api/comfyui-workflow";
import {
  COMFYUI_MEDIA_TYPE_OPTIONS,
  COMFYUI_OUTPUT_ROLE_OPTIONS,
  COMFYUI_VALUE_TYPE_OPTIONS,
  flattenInputBindings,
  getComfyUiBusinessFields,
  groupInputBindings,
  parseComfyUiNodes,
  type FlatComfyUiInputBinding,
} from "./comfyui-workflow-support";

interface ComfyUiBindingEditorProps {
  mode: "input" | "output";
  modelType: ComfyUiWorkflowModelType;
  apiWorkflowJson: string;
  inputBindings: ComfyUiInputBindings;
  outputBindings: ComfyUiOutputBinding[];
  onInputBindingsChange: (value: ComfyUiInputBindings) => void;
  onOutputBindingsChange: (value: ComfyUiOutputBinding[]) => void;
}

function OptionSelect({
  value,
  options,
  placeholder,
  onChange,
}: {
  value?: string;
  options: ReadonlyArray<{ value: string; label: string }>;
  placeholder: string;
  onChange: (value: string) => void;
}) {
  return (
    <Select
      value={value || undefined}
      onValueChange={next => onChange(String(next))}
      items={options.map(option => ({ value: option.value, label: option.label }))}
    >
      <SelectTrigger className="w-full text-xs">
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent className="text-xs">
        <SelectGroup>
          {options.map(option => (
            <SelectItem key={option.value} value={option.value} className="text-xs">
              {option.label}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}

export function ComfyUiBindingEditor({
  mode,
  modelType,
  apiWorkflowJson,
  inputBindings,
  outputBindings,
  onInputBindingsChange,
  onOutputBindingsChange,
}: ComfyUiBindingEditorProps) {
  let nodes = [] as ReturnType<typeof parseComfyUiNodes>;
  let parseError = "";
  try {
    nodes = parseComfyUiNodes(apiWorkflowJson);
  } catch (error) {
    parseError = error instanceof Error ? error.message : "工作流 JSON 解析失败";
  }

  const nodeOptions = nodes.map(node => ({
    value: node.id,
    label: `${node.id} · ${node.classType}`,
  }));
  const inputRows = flattenInputBindings(inputBindings);

  const updateInputRow = (index: number, patch: Partial<FlatComfyUiInputBinding>) => {
    const rows = inputRows.map((row, rowIndex) => {
      if (rowIndex !== index) return row;
      const next = { ...row, ...patch };
      if (patch.nodeId !== undefined) {
        next.inputName = nodes.find(node => node.id === patch.nodeId)?.inputNames[0] || "";
      }
      return next;
    });
    onInputBindingsChange(groupInputBindings(rows));
  };

  const addInputRow = () => {
    const node = nodes[0];
    onInputBindingsChange(groupInputBindings([
      ...inputRows,
      {
        businessField: getComfyUiBusinessFields(modelType)[0],
        nodeId: node?.id || "",
        inputName: node?.inputNames[0] || "",
        valueType: "string",
      },
    ]));
  };

  const addOutputRow = () => {
    onOutputBindingsChange([
      ...outputBindings,
      {
        nodeId: nodes[0]?.id || "",
        mediaType: modelType === 2 ? "image" : "video",
        role: outputBindings.length === 0 ? "primary" : "auxiliary",
      },
    ]);
  };

  if (parseError || nodes.length === 0) {
    return (
      <div className="rounded-lg border border-border/20 bg-background/70 p-4 text-sm text-muted-foreground">
        {parseError || "请先导入有效的 API-format 工作流，随后再配置绑定。"}
      </div>
    );
  }

  if (mode === "input") {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-sm font-medium">平台输入绑定</p>
            <p className="mt-1 text-xs text-muted-foreground">把业务字段写入明确的节点输入；一个业务字段可绑定多个目标。</p>
          </div>
          <Button variant="outline" size="sm" onClick={addInputRow}>
            <Plus />添加绑定
          </Button>
        </div>

        {inputRows.length === 0 ? (
          <div className="rounded-lg border border-border/20 bg-background/70 p-4 text-sm text-muted-foreground">
            尚未配置输入绑定。
          </div>
        ) : inputRows.map((row, index) => {
          const node = nodes.find(item => item.id === row.nodeId);
          return (
            <div key={`${row.businessField}-${row.nodeId}-${row.inputName}-${index}`} className="rounded-lg border border-border/20 bg-background/70 p-3">
              <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-[1fr_1.4fr_1fr_1fr_80px_36px]">
                <div className="space-y-1.5">
                  <Label className="text-[11px] text-muted-foreground">平台字段</Label>
                  <OptionSelect
                    value={row.businessField}
                    options={getComfyUiBusinessFields(modelType).map(value => ({ value, label: value }))}
                    placeholder="选择字段"
                    onChange={businessField => updateInputRow(index, { businessField })}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[11px] text-muted-foreground">目标节点</Label>
                  <OptionSelect value={row.nodeId} options={nodeOptions} placeholder="选择节点" onChange={nodeId => updateInputRow(index, { nodeId })} />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[11px] text-muted-foreground">节点输入</Label>
                  <OptionSelect
                    value={row.inputName}
                    options={(node?.inputNames || []).map(value => ({ value, label: value }))}
                    placeholder="选择输入"
                    onChange={inputName => updateInputRow(index, { inputName })}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[11px] text-muted-foreground">值类型</Label>
                  <OptionSelect
                    value={row.valueType}
                    options={COMFYUI_VALUE_TYPE_OPTIONS}
                    placeholder="选择类型"
                    onChange={valueType => updateInputRow(index, { valueType: valueType as ComfyUiBindingValueType })}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-[11px] text-muted-foreground">索引</Label>
                  <Input
                    type="number"
                    min={0}
                    value={row.index ?? ""}
                    placeholder="可选"
                    onChange={event => updateInputRow(index, { index: event.target.value ? Number(event.target.value) : undefined })}
                    className="text-xs"
                  />
                </div>
                <div className="flex items-end">
                  <Button
                    variant="destructive-ghost"
                    size="icon"
                    aria-label="删除输入绑定"
                    onClick={() => onInputBindingsChange(groupInputBindings(inputRows.filter((_, rowIndex) => rowIndex !== index)))}
                  >
                    <Trash2 />
                  </Button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium">结果输出绑定</p>
          <p className="mt-1 text-xs text-muted-foreground">图片工作流必须有 primary image；视频工作流必须有 primary video。</p>
        </div>
        <Button variant="outline" size="sm" onClick={addOutputRow}>
          <Plus />添加输出
        </Button>
      </div>

      {outputBindings.length === 0 ? (
        <div className="rounded-lg border border-border/20 bg-background/70 p-4 text-sm text-muted-foreground">尚未配置输出绑定。</div>
      ) : outputBindings.map((row, index) => (
        <div key={`${row.nodeId}-${row.mediaType}-${row.role}-${index}`} className="rounded-lg border border-border/20 bg-background/70 p-3">
          <div className="grid gap-2 md:grid-cols-[1.5fr_1fr_1fr_36px]">
            <div className="space-y-1.5">
              <Label className="text-[11px] text-muted-foreground">输出节点</Label>
              <OptionSelect
                value={row.nodeId}
                options={nodeOptions}
                placeholder="选择节点"
                onChange={nodeId => onOutputBindingsChange(outputBindings.map((item, rowIndex) => rowIndex === index ? { ...item, nodeId } : item))}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-[11px] text-muted-foreground">媒体类型</Label>
              <OptionSelect
                value={row.mediaType}
                options={COMFYUI_MEDIA_TYPE_OPTIONS}
                placeholder="选择媒体"
                onChange={mediaType => onOutputBindingsChange(outputBindings.map((item, rowIndex) => rowIndex === index ? { ...item, mediaType: mediaType as ComfyUiOutputMediaType } : item))}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-[11px] text-muted-foreground">结果角色</Label>
              <OptionSelect
                value={row.role}
                options={COMFYUI_OUTPUT_ROLE_OPTIONS}
                placeholder="选择角色"
                onChange={role => onOutputBindingsChange(outputBindings.map((item, rowIndex) => rowIndex === index ? { ...item, role: role as ComfyUiOutputRole } : item))}
              />
            </div>
            <div className="flex items-end">
              <Button variant="destructive-ghost" size="icon" aria-label="删除输出绑定" onClick={() => onOutputBindingsChange(outputBindings.filter((_, rowIndex) => rowIndex !== index))}>
                <Trash2 />
              </Button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
