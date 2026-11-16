"use client";

import { useId } from "react";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ComfyUiBindingSelect } from "./comfyui-binding-select";
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
  formatComfyUiNodeLabel,
  getComfyUiBusinessFields,
  groupInputBindings,
  isComfyUiMediaField,
  parseComfyUiNodes,
  updateComfyUiInputBinding,
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

export function ComfyUiBindingEditor({
  mode,
  modelType,
  apiWorkflowJson,
  inputBindings,
  outputBindings,
  onInputBindingsChange,
  onOutputBindingsChange,
}: ComfyUiBindingEditorProps) {
  const id = useId();
  let nodes = [] as ReturnType<typeof parseComfyUiNodes>;
  let parseError = "";
  try {
    nodes = parseComfyUiNodes(apiWorkflowJson);
  } catch (error) {
    parseError =
      error instanceof Error ? error.message : "工作流 JSON 解析失败";
  }

  const nodeOptions = nodes.map((node) => ({
    value: node.id,
    label: formatComfyUiNodeLabel(node),
  }));
  const inputRows = flattenInputBindings(inputBindings);

  const updateInputRow = (
    index: number,
    patch: Partial<FlatComfyUiInputBinding>,
  ) => {
    const rows = inputRows.map((row, rowIndex) => {
      if (rowIndex !== index) {
        return row;
      }
      return updateComfyUiInputBinding(row, patch, nodes);
    });
    onInputBindingsChange(groupInputBindings(rows));
  };

  const addInputRow = () => {
    const node = nodes[0];
    onInputBindingsChange(
      groupInputBindings([
        ...inputRows,
        {
          businessField: getComfyUiBusinessFields(modelType)[0],
          nodeId: node?.id || "",
          inputName: node?.inputNames[0] || "",
          valueType: "string",
        },
      ]),
    );
  };

  const addOutputRow = () => {
    onOutputBindingsChange([
      ...outputBindings,
      {
        nodeId: "",
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
        <div className="flex flex-col items-start gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <p className="text-sm font-medium">平台输入绑定</p>
            <p className="mt-1 text-xs text-muted-foreground">
              把业务字段写入明确的节点输入；一个业务字段可绑定多个目标。
            </p>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              多图输入请从索引 0 开始逐一绑定，可用数量以工作流实际配置为准。
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={addInputRow}>
            <Plus />
            添加绑定
          </Button>
        </div>

        {inputRows.length === 0 ? (
          <div className="rounded-lg border border-border/20 bg-background/70 p-4 text-sm text-muted-foreground">
            尚未配置输入绑定。
          </div>
        ) : (
          inputRows.map((row, index) => {
            const node = nodes.find((item) => item.id === row.nodeId);
            return (
              <div
                key={`${row.businessField}-${row.nodeId}-${row.inputName}-${index}`}
                className="rounded-lg border border-border/20 bg-background/70 p-3"
              >
                <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-[1fr_1.4fr_1fr_1fr_80px_36px]">
                  <div className="flex min-w-0 flex-col gap-2">
                    <Label className="text-[11px] text-muted-foreground">
                      平台字段
                    </Label>
                    <ComfyUiBindingSelect
                      value={row.businessField}
                      options={getComfyUiBusinessFields(modelType).map(
                        (value) => ({ value, label: value }),
                      )}
                      placeholder="选择字段"
                      onChange={(businessField) =>
                        updateInputRow(index, { businessField })
                      }
                    />
                  </div>
                  <div className="flex min-w-0 flex-col gap-2">
                    <Label className="text-[11px] text-muted-foreground">
                      目标节点
                    </Label>
                    <ComfyUiBindingSelect
                      value={row.nodeId}
                      options={nodeOptions}
                      placeholder="选择节点"
                      onChange={(nodeId) => updateInputRow(index, { nodeId })}
                    />
                  </div>
                  <div className="flex min-w-0 flex-col gap-2">
                    <Label className="text-[11px] text-muted-foreground">
                      节点输入
                    </Label>
                    <ComfyUiBindingSelect
                      value={row.inputName}
                      options={(node?.inputNames || []).map((value) => ({
                        value,
                        label: value,
                      }))}
                      placeholder="选择输入"
                      onChange={(inputName) =>
                        updateInputRow(index, { inputName })
                      }
                    />
                  </div>
                  <div className="flex min-w-0 flex-col gap-2">
                    <Label className="text-[11px] text-muted-foreground">
                      值类型
                    </Label>
                    <ComfyUiBindingSelect
                      value={row.valueType}
                      options={COMFYUI_VALUE_TYPE_OPTIONS}
                      placeholder="选择类型"
                      onChange={(valueType) =>
                        updateInputRow(index, {
                          valueType: valueType as ComfyUiBindingValueType,
                        })
                      }
                    />
                  </div>
                  <div className="flex min-w-0 flex-col gap-2">
                    <Label className="text-[11px] text-muted-foreground">
                      索引
                    </Label>
                    <Input
                      type="number"
                      min={0}
                      value={row.index ?? ""}
                      placeholder="可选"
                      onChange={(event) =>
                        updateInputRow(index, {
                          index: event.target.value
                            ? Number(event.target.value)
                            : undefined,
                        })
                      }
                      className="text-xs"
                    />
                  </div>
                  <div className="flex items-end">
                    <Button
                      variant="destructive-ghost"
                      size="icon"
                      aria-label="删除输入绑定"
                      onClick={() =>
                        onInputBindingsChange(
                          groupInputBindings(
                            inputRows.filter(
                              (_, rowIndex) => rowIndex !== index,
                            ),
                          ),
                        )
                      }
                    >
                      <Trash2 />
                    </Button>
                  </div>
                </div>
                {isComfyUiMediaField(row.businessField) && (
                  <div className="mt-3 flex flex-wrap items-center gap-2 p-1">
                    <Checkbox
                      id={`${id}-optional-${index}`}
                      checked={row.optional === true}
                      aria-label={`${row.businessField} · 节点 ${row.nodeId} 的 ${row.inputName} 可选`}
                      aria-describedby={`${id}-optional-help-${index}`}
                      onCheckedChange={(checked) =>
                        updateInputRow(index, { optional: checked === true })
                      }
                    />
                    <Label htmlFor={`${id}-optional-${index}`}>可选</Label>
                    <p
                      id={`${id}-optional-help-${index}`}
                      className="text-xs text-muted-foreground"
                    >
                      未提供时断开此媒体输入
                    </p>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium">结果输出绑定</p>
          <p className="mt-1 text-xs text-muted-foreground">
            选择会向 ComfyUI 任务历史写入文件的最终节点，如 SaveImage
            或视频合成/保存节点；不要选择采样、解码等中间节点。
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={addOutputRow}>
          <Plus />
          添加输出
        </Button>
      </div>

      <div className="rounded-lg border border-border/20 bg-background/70 p-3 text-xs text-muted-foreground">
        平台会从所选节点返回的文件中按媒体类型提取结果，不需要选择节点连线上的输出插槽。主结果用于生成成品；视频封面仅用于视频任务的封面图；附加结果不会作为主图或主视频。
      </div>

      {outputBindings.length === 0 ? (
        <div className="rounded-lg border border-border/20 bg-background/70 p-4 text-sm text-muted-foreground">
          尚未配置输出绑定。
        </div>
      ) : (
        outputBindings.map((row, index) => (
          <div
            key={`${row.nodeId}-${row.mediaType}-${row.role}-${index}`}
            className="rounded-lg border border-border/20 bg-background/70 p-3"
          >
            <div className="grid gap-2 md:grid-cols-[1.5fr_1fr_1fr_36px]">
              <div className="flex min-w-0 flex-col gap-2">
                <Label className="text-[11px] text-muted-foreground">
                  文件输出节点
                </Label>
                <ComfyUiBindingSelect
                  value={row.nodeId}
                  options={nodeOptions}
                  placeholder="选择保存结果的节点"
                  onChange={(nodeId) =>
                    onOutputBindingsChange(
                      outputBindings.map((item, rowIndex) =>
                        rowIndex === index ? { ...item, nodeId } : item,
                      ),
                    )
                  }
                />
              </div>
              <div className="flex min-w-0 flex-col gap-2">
                <Label className="text-[11px] text-muted-foreground">
                  提取媒体类型
                </Label>
                <ComfyUiBindingSelect
                  value={row.mediaType}
                  options={COMFYUI_MEDIA_TYPE_OPTIONS.filter((option) => {
                    if (row.role === "primary")
                      return (
                        option.value === (modelType === 2 ? "image" : "video")
                      );
                    if (row.role === "cover") return option.value === "image";
                    return true;
                  })}
                  placeholder="选择媒体"
                  onChange={(mediaType) =>
                    onOutputBindingsChange(
                      outputBindings.map((item, rowIndex) =>
                        rowIndex === index
                          ? {
                              ...item,
                              mediaType: mediaType as ComfyUiOutputMediaType,
                            }
                          : item,
                      ),
                    )
                  }
                />
              </div>
              <div className="flex min-w-0 flex-col gap-2">
                <Label className="text-[11px] text-muted-foreground">
                  结果用途
                </Label>
                <ComfyUiBindingSelect
                  value={row.role}
                  options={COMFYUI_OUTPUT_ROLE_OPTIONS.filter(
                    (option) => modelType === 3 || option.value !== "cover",
                  )}
                  placeholder="选择用途"
                  onChange={(role) =>
                    onOutputBindingsChange(
                      outputBindings.map((item, rowIndex) => {
                        if (rowIndex !== index) return item;
                        const nextRole = role as ComfyUiOutputRole;
                        const requiredMediaType =
                          nextRole === "primary"
                            ? modelType === 2
                              ? "image"
                              : "video"
                            : nextRole === "cover"
                              ? "image"
                              : item.mediaType;
                        return {
                          ...item,
                          role: nextRole,
                          mediaType: requiredMediaType,
                        };
                      }),
                    )
                  }
                />
              </div>
              <div className="flex items-end">
                <Button
                  variant="destructive-ghost"
                  size="icon"
                  aria-label="删除输出绑定"
                  onClick={() =>
                    onOutputBindingsChange(
                      outputBindings.filter(
                        (_, rowIndex) => rowIndex !== index,
                      ),
                    )
                  }
                >
                  <Trash2 />
                </Button>
              </div>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
