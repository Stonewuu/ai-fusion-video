"use client";

import { useEffect, useId, useState } from "react";
import { aiModelApi, type ComfyUiResolvedCapabilities } from "@/lib/api/ai-model";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { isConfigRecord, parseConfigJson, stringifyConfigObject } from "./model-config-support";

const INPUTS = [
  { key: "referenceImages", label: "参考图", maximum: "maxReferenceImages", support: "supportReferenceImages", unit: "张" },
  { key: "firstFrame", label: "首帧", maximum: "maxFirstFrames", support: "supportFirstFrame", unit: "张" },
  { key: "lastFrame", label: "尾帧", maximum: "maxLastFrames", support: "supportLastFrame", unit: "张" },
  { key: "referenceVideos", label: "参考视频", maximum: "maxReferenceVideos", support: "supportReferenceVideos", unit: "个" },
  { key: "referenceAudios", label: "参考音频", maximum: "maxReferenceAudios", support: "supportReferenceAudios", unit: "个" },
  { key: "imageInputs", label: "图片输入总数", maximum: "maxImageInputs", unit: "张" },
] as const;

function savedLimit(config: Record<string, unknown>, field: (typeof INPUTS)[number]): number | null {
  const limits = isConfigRecord(config.comfyuiInputLimits) ? config.comfyuiInputLimits : {};
  if (Object.hasOwn(limits, field.key)) {
    return limits[field.key] === null ? null : Number(limits[field.key]);
  }
  if ("support" in field && config[field.support] === false) {
    return 0;
  }
  const legacy = field.key === "referenceImages" ? config.maxRefImages
    : field.key === "imageInputs" ? config.maxImages : undefined;
  const value = config[field.maximum] ?? legacy;
  return value === undefined || value === null ? null : Number(value);
}

export function ComfyUiInputLimits({ apiConfigId, modelType, workflowId, configJson, capabilityPresetCode, configuredLimits, onChange, onValidityChange }: {
  apiConfigId: number;
  modelType: 2 | 3;
  workflowId: number;
  configJson?: string;
  capabilityPresetCode?: string;
  configuredLimits?: Record<string, unknown>;
  onChange: (json: string) => void;
  onValidityChange: (valid: boolean) => void;
}) {
  const id = useId();
  const sourceKey = JSON.stringify([apiConfigId, modelType, workflowId]);
  const requestKey = JSON.stringify([apiConfigId, modelType, workflowId, configJson, capabilityPresetCode]);
  const [result, setResult] = useState<{ key: string; source: string; data?: ComfyUiResolvedCapabilities; error?: string } | null>(null);
  const [inputError, setInputError] = useState<string | null>(null);
  const current = result?.key === requestKey ? result : null;
  // 重新校验使用限制时保持字段位置；仅有效的新响应可以允许保存。
  const workflow = result?.source === sourceKey ? result.data : undefined;
  const config = parseConfigJson(configJson);
  const savedConfig = configuredLimits ?? config;

  useEffect(() => {
    let cancelled = false;
    void aiModelApi.comfyUiCapabilities({
      apiConfigId, modelType, comfyuiWorkflowId: workflowId, config: configJson, capabilityPresetCode,
    }).then((data) => {
      if (!cancelled) {
        setResult({ key: requestKey, source: sourceKey, data });
        onValidityChange(true);
      }
    }).catch((error) => {
      if (!cancelled) {
        onValidityChange(false);
        setResult((previous) => ({ key: requestKey, source: sourceKey,
          data: previous?.source === sourceKey ? previous.data : undefined, error: getApiErrorMessage(error) }));
      }
    });
    return () => {
      cancelled = true;
    };
  }, [apiConfigId, modelType, workflowId, configJson, capabilityPresetCode, requestKey, sourceKey, onValidityChange]);

  const updateLimit = (field: string, raw: string) => {
    const value = raw.trim() ? Number(raw) : null;
    if (value !== null && (!Number.isSafeInteger(value) || value < 0)) {
      onValidityChange(false);
      setInputError("使用上限须为非负整数，留空表示跟随工作流。");
      return;
    }
    setInputError(null);
    const limits = isConfigRecord(config.comfyuiInputLimits) ? config.comfyuiInputLimits : {};
    const nextConfig = stringifyConfigObject({ ...config, comfyuiInputLimits: { ...limits, [field]: value } });
    if (nextConfig === configJson) {
      onValidityChange(Boolean(current?.data && !current.error));
    }
    onChange(nextConfig);
  };

  const visible = INPUTS.filter((field) => {
    if (workflow) {
      return field.key === "imageInputs"
        ? modelType === 3 && ["firstFrame", "lastFrame", "referenceImages"].some((key) => workflow.comfyuiInputs[key])
        : Boolean(workflow.comfyuiInputs[field.key]);
    }
    return field.key === "referenceImages" || savedLimit(savedConfig, field) !== null;
  });

  return (
    <div className="flex min-w-0 flex-col gap-3 border-t border-border/20 pt-3">
      <div className="flex flex-col gap-1">
        <p className="text-xs font-medium">输入使用上限</p>
        <p className="text-xs leading-5 text-muted-foreground">
          {current?.data ? `能力来自已发布版本 v${current.data.comfyuiWorkflowVersionNo}。` : !current ? "正在读取工作流能力。" : ""}
          留空跟随工作流，0 表示不允许此输入。
        </p>
      </div>
      {(current?.error || inputError) && (
        <p role="alert" className="text-xs leading-5 text-destructive">{inputError || current?.error}</p>
      )}
      {visible.map((field) => {
        const contract = workflow?.comfyuiInputs[field.key];
        const value = current?.data && !current.error
          ? field.key === "imageInputs" ? current.data.comfyuiImageUsageMaximum : contract?.usageMaximum ?? null
          : savedLimit(savedConfig, field);
        return (
          <div key={field.key} className="flex min-w-0 flex-col gap-2 p-1">
            <Label htmlFor={`${id}-${field.key}`} className="text-xs">{field.label}</Label>
            <div className="flex flex-wrap items-center gap-2">
              <Input
                key={`${field.key}-${value}`}
                id={`${id}-${field.key}`}
                type="number"
                min={0}
                step={1}
                disabled={!current}
                aria-invalid={Boolean(inputError)}
                defaultValue={value ?? ""}
                placeholder="跟随工作流"
                className="min-w-0 flex-1"
                onBlur={(event) => {
                  const raw = event.currentTarget.value;
                  if (inputError || (raw.trim() ? Number(raw) : null) !== value) {
                    updateLimit(field.key, raw);
                  }
                }}
              />
              <Button variant="outline" disabled={!current || value === null && !current.error} onClick={() => updateLimit(field.key, "")}>
                跟随工作流
              </Button>
            </div>
            {contract && (
              <p className="text-xs leading-5 text-muted-foreground">
                {contract.workflowMaximum === null ? "工作流接收列表。" : `工作流可接收 ${contract.workflowMaximum} ${field.unit}。`}
                {contract.minimumWhenProvided > 1 ? `提供此输入时，需传入 ${contract.minimumWhenProvided} ${field.unit}。` : ""}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
