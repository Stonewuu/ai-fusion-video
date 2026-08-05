"use client";

import { useEffect, useState } from "react";
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
import { toastApiError } from "@/lib/api/toast-api-error";
import {
  comfyUiWorkflowApi,
  type ComfyUiWorkflow,
  type ComfyUiWorkflowModelType,
} from "@/lib/api/comfyui-workflow";
import { parseConfigJson, stringifyConfigObject } from "./model-config-support";

interface ComfyUiModelWorkflowFieldProps {
  apiConfigId: number;
  modelType: ComfyUiWorkflowModelType;
  value?: number;
  configJson?: string;
  onChange: (workflowId: number, configJson: string) => void;
}

function positiveConfigNumber(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function ComfyUiModelWorkflowField({
  apiConfigId,
  modelType,
  value,
  configJson,
  onChange,
}: ComfyUiModelWorkflowFieldProps) {
  const [workflows, setWorkflows] = useState<ComfyUiWorkflow[]>([]);
  const [loadedKey, setLoadedKey] = useState("");
  const requestKey = `${apiConfigId}:${modelType}`;
  const loading = loadedKey !== requestKey;
  const visibleWorkflows = loading ? [] : workflows;
  const config = parseConfigJson(configJson);
  const defaultPollIntervalMillis = modelType === 2 ? 2_000 : 5_000;
  const defaultTimeoutMillis = modelType === 2 ? 25 * 60_000 : 60 * 60_000;
  const pollIntervalSeconds = positiveConfigNumber(
    config.comfyuiPollIntervalMillis,
    defaultPollIntervalMillis,
  ) / 1_000;
  const timeoutMinutes = positiveConfigNumber(
    config.comfyuiTimeoutMillis,
    defaultTimeoutMillis,
  ) / 60_000;

  const updateRuntimeConfig = (key: "comfyuiPollIntervalMillis" | "comfyuiTimeoutMillis", nextValue: number) => {
    if (!value) return;
    const next = parseConfigJson(configJson);
    next[key] = Math.round(nextValue);
    onChange(value, stringifyConfigObject(next));
  };

  useEffect(() => {
    let cancelled = false;
    comfyUiWorkflowApi.list(apiConfigId, modelType)
      .then(items => {
        if (!cancelled) setWorkflows(items);
      })
      .catch(error => {
        if (!cancelled) {
          setWorkflows([]);
          toastApiError(error, "加载 ComfyUI 工作流失败");
        }
      })
      .finally(() => {
        if (!cancelled) setLoadedKey(requestKey);
      });
    return () => { cancelled = true; };
  }, [apiConfigId, modelType, requestKey]);

  const selectWorkflow = async (workflowId: number) => {
    const workflow = visibleWorkflows.find(item => item.id === workflowId);
    if (!workflow?.activeVersionId) return;
    try {
      const version = await comfyUiWorkflowApi.getVersion(workflow.activeVersionId);
      const bindings = JSON.parse(version.inputBindingsJson) as Record<string, Array<{ index?: number }>>;
      const hasField = (field: string) => Array.isArray(bindings[field]) && bindings[field].length > 0;
      const maxInputs = (field: string) => Math.max(
        1,
        ...(bindings[field] || []).map(item => (item.index ?? 0) + 1),
      );
      const config = parseConfigJson(configJson);
      if (modelType === 2) {
        const supportsReferences = hasField("referenceImages");
        Object.assign(config, {
          supportReferenceImages: supportsReferences,
          minReferenceImages: 0,
          maxReferenceImages: supportsReferences ? maxInputs("referenceImages") : 0,
          referenceImageInputFormats: supportsReferences ? ["url", "data_uri"] : [],
          supportDataUriInput: supportsReferences,
        });
      } else {
        const supportsFirstFrame = hasField("firstFrame");
        const supportsLastFrame = hasField("lastFrame");
        const supportsReferenceImages = hasField("referenceImages");
        const supportsAnyImage = supportsFirstFrame || supportsLastFrame || supportsReferenceImages;
        Object.assign(config, {
          supportFirstFrame: supportsFirstFrame,
          supportLastFrame: supportsLastFrame,
          supportReferenceImages: supportsReferenceImages,
          maxReferenceImages: supportsReferenceImages ? maxInputs("referenceImages") : 0,
          supportReferenceVideos: false,
          maxReferenceVideos: 0,
          supportReferenceAudios: false,
          maxReferenceAudios: 0,
          referenceImageInputFormats: supportsAnyImage ? ["url", "data_uri"] : [],
          supportDataUriInput: supportsAnyImage,
        });
      }
      onChange(workflowId, stringifyConfigObject(config));
    } catch (error) {
      toastApiError(error, "读取 ComfyUI 工作流能力失败");
    }
  };

  return (
    <div className="rounded-xl border border-primary/20 bg-primary/[0.035] p-3 space-y-2">
      <div>
        <Label className="text-xs font-medium text-foreground">已发布 ComfyUI 工作流</Label>
        <p className="mt-1 text-[11px] leading-4 text-muted-foreground">
          任务提交时会固定当前发布版本；后续发布新版本不会改变已排队任务。
        </p>
      </div>
      <Select
        value={value ?? null}
        onValueChange={next => {
          if (next !== null) void selectWorkflow(Number(next));
        }}
        items={visibleWorkflows.map(workflow => ({ value: workflow.id, label: `${workflow.name} (${workflow.code})` }))}
      >
        <SelectTrigger className="w-full text-sm">
          <SelectValue placeholder={loading ? "正在加载工作流" : "选择已发布工作流"} />
        </SelectTrigger>
        <SelectContent className="text-sm">
          <SelectGroup>
            {visibleWorkflows.map(workflow => (
              <SelectItem key={workflow.id} value={workflow.id} className="text-sm">
                {workflow.name}<span className="ml-1 font-mono text-xs text-muted-foreground">{workflow.code}</span>
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
      {!loading && visibleWorkflows.length === 0 && (
        <p className="text-[10px] leading-4 text-destructive">
          当前实例没有同类型且已发布的工作流，请先在工作流管理中完成验证、试运行和发布。
        </p>
      )}
      {value && (
        <div className="space-y-3 border-t border-primary/10 pt-3">
          <div>
            <p className="text-xs font-medium text-foreground">任务执行参数</p>
            <p className="mt-1 text-[11px] leading-4 text-muted-foreground">
              仅影响使用此模型发起的生成任务；工作流试运行仍使用系统固定参数。
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="comfyui-poll-interval" className="text-[11px] text-muted-foreground">轮询间隔（秒）</Label>
              <Input
                key={pollIntervalSeconds}
                id="comfyui-poll-interval"
                type="number"
                min={0.5}
                step={0.5}
                defaultValue={pollIntervalSeconds}
                className="font-mono"
                onBlur={event => {
                  const parsed = Number(event.currentTarget.value);
                  const seconds = Number.isFinite(parsed) ? Math.max(0.5, parsed) : pollIntervalSeconds;
                  event.currentTarget.value = String(seconds);
                  updateRuntimeConfig("comfyuiPollIntervalMillis", seconds * 1_000);
                }}
                onKeyDown={event => {
                  if (event.key === "Enter") event.currentTarget.blur();
                }}
              />
              <p className="text-[10px] leading-4 text-muted-foreground">查询 ComfyUI 任务状态的频率，最短 0.5 秒。</p>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="comfyui-timeout" className="text-[11px] text-muted-foreground">任务超时（分钟）</Label>
              <Input
                key={timeoutMinutes}
                id="comfyui-timeout"
                type="number"
                min={1}
                step={1}
                defaultValue={timeoutMinutes}
                className="font-mono"
                onBlur={event => {
                  const parsed = Number(event.currentTarget.value);
                  const minutes = Number.isFinite(parsed) ? Math.max(1, parsed) : timeoutMinutes;
                  event.currentTarget.value = String(minutes);
                  updateRuntimeConfig("comfyuiTimeoutMillis", minutes * 60_000);
                }}
                onKeyDown={event => {
                  if (event.key === "Enter") event.currentTarget.blur();
                }}
              />
              <p className="text-[10px] leading-4 text-muted-foreground">超过该时长仍未完成时，将任务标记为超时。</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
