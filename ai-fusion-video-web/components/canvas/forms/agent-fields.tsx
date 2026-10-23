"use client";

import { useId } from "react";
import { ImagePlus, Clapperboard } from "lucide-react";
import { CanvasNumberField } from "./field-controls";
import { canvasTypeNames } from "@/lib/canvas/labels";
import type { CanvasCapability, CanvasConfig, CanvasValueType } from "@/lib/canvas/types";
import { useCanvas } from "../canvas-provider";
import { CanvasSegmentedControl } from "../workspace/segmented-control";
import { CanvasSelect } from "./canvas-select";
import { CanvasAgentSkillFields } from "./agent-skill-fields";

export function CanvasAgentFields({ capability, config, onChange, disabled, connectedPorts }: {
  capability: CanvasCapability;
  config: CanvasConfig;
  onChange: (value: CanvasConfig) => void;
  disabled: boolean;
  connectedPorts: string[];
}) {
  const formId = useId();
  const models = useCanvas((state) => state.catalog!.models);
  const profiles = useCanvas((state) => state.catalog!.agentProfiles);
  const type = typeof config.agentType === "string" ? config.agentType : "";
  const profile = profiles.find((item) => item.type === type);
  const update = (field: string, value: string | number | undefined) => {
    const next = { ...config };
    if (value === undefined) delete next[field]; else next[field] = value;
    onChange(next);
  };
  return (
    <section className="flex flex-col gap-4" aria-label="专业智能体设置">
      {capability.kind === "agent" ? (
        <div className="flex flex-col gap-2">
          <span className="text-sm font-medium">专业智能体</span>
          <CanvasSelect label="专业智能体" value={type || null} placeholder="选择创作能力" disabled={disabled}
            items={profiles.filter((item) => !connectedPorts.includes("inputs") || item.inputTypes.includes(config.inputType as CanvasValueType))
              .map((item) => ({ value: item.type, label: item.name }))}
            onChange={(value) => {
              const selected = profiles.find((item) => item.type === value)!;
              const next: CanvasConfig = { ...config, agentType: value, maxImages: 0, maxVideoSeconds: 0 };
              delete next.imageModelId;
              delete next.videoModelId;
              if (!value.startsWith("storyboard_frame")) delete next.frameType;
              else if (!next.frameType) next.frameType = "first";
              if (!selected.inputTypes.includes(config.inputType as CanvasValueType)) next.inputType = selected.inputTypes[0];
              if (!selected.resultTypes.includes(config.resultType as CanvasValueType)) next.resultType = selected.resultTypes[0];
              onChange(next);
            }} />
        </div>
      ) : null}
      {profile ? <p className="text-sm text-muted-foreground">{profile.hint}</p> : null}
      {profile && !profile.toolsEnabled && <p className="text-sm text-destructive">此智能体已关闭工具调用，只能生成文字。业务结果需要在智能体配置中开启工具。</p>}
      <p className="text-xs text-muted-foreground">可从节点的参考端口同时接入图片、视频、音频和文档；运行前会检查所选模型是否支持。</p>
      {capability.kind === "agent" && profile ? (
        <div className="grid gap-4 sm:grid-cols-2">
          {(["inputType", "resultType"] as const).map((field) => (
            <div key={field} className="flex flex-col gap-2">
              <span className="text-sm font-medium">{field === "inputType" ? "主要输入类型" : "输出结果类型"}</span>
              <CanvasSelect label={field === "inputType" ? "主要输入类型" : "输出结果类型"}
                value={typeof config[field] === "string" ? config[field] : null} placeholder="选择内容类型"
                disabled={disabled || field === "inputType" && connectedPorts.includes("inputs")}
                items={(field === "inputType" ? profile.inputTypes : profile.resultTypes).map((value) => ({ value, label: canvasTypeNames[value] }))}
                onChange={(value) => update(field, value)} />
            </div>
          ))}
        </div>
      ) : null}
      <details className="border-t border-border/20 pt-4">
        <summary className="cursor-pointer rounded-lg text-sm font-medium outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">工作规范与执行上限</summary>
        <div className="flex flex-col gap-4 pt-4">
      <CanvasAgentSkillFields profile={profile} config={config} onChange={onChange} disabled={disabled} />
      {(["image", "video"] as const).filter((kind) => capability.configFields[kind === "image" ? "maxImages" : "maxVideoSeconds"] &&
        (profile?.generationKinds.includes(kind) || Number(config[kind === "image" ? "maxImages" : "maxVideoSeconds"]) > 0)).map((kind) => {
        const budget = kind === "image" ? "maxImages" : "maxVideoSeconds";
        const model = kind === "image" ? "imageModelId" : "videoModelId";
        const label = kind === "image" ? "图片生成" : "视频生成";
        const enabled = config[budget] !== undefined && config[budget] !== 0;
        const Icon = kind === "image" ? ImagePlus : Clapperboard;
        const options = models.filter((item) => item.kind === kind && !item.unavailable);
        return (
          <div key={kind} className="flex flex-col gap-3 rounded-lg border border-border/20 bg-background/70 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="flex items-center gap-2 text-sm font-medium"><Icon className="size-4 text-primary" />{label}</span>
              <CanvasSegmentedControl label={`允许${label}`} value={enabled ? "allow" : "off"}
                options={[{ value: "off", label: "关闭", disabled }, { value: "allow", label: "允许", disabled }]}
                onChange={(value) => {
                  const next = { ...config, [budget]: value === "off" ? 0 : kind === "image" ? 1 : 5 };
                  if (value === "off") delete next[model];
                  onChange(next);
                }} />
            </div>
            {enabled ? (
              <>
                <div className="flex flex-col gap-2">
                  <span className="text-sm font-medium">{label}模型</span>
                  <CanvasSelect label={`${label}模型`} value={typeof config[model] === "number" ? String(config[model]) : null}
                    placeholder={options.length ? "选择本次使用的模型" : "尚未配置可用模型"} disabled={disabled}
                    items={options.map((item) => ({ value: String(item.id), label: item.name }))}
                    onChange={(value) => update(model, Number(value))} />
                </div>
                <div className="flex flex-col gap-2">
                  <label htmlFor={`${formId}-${budget}`} className="text-sm font-medium">{kind === "image" ? "最多生成图片（张）" : "最多生成视频（秒）"}</label>
                  <CanvasNumberField id={`${formId}-${budget}`} label={kind === "image" ? "最多生成图片" : "最多生成视频"} min={1} max={kind === "image" ? 1000 : 36000}
                    value={typeof config[budget] === "number" ? config[budget] as number : undefined} disabled={disabled}
                    presets={kind === "image" ? [1, 4, 8] : [5, 15, 30]} unit={kind === "image" ? "张" : "秒"}
                    onChange={(value) => update(budget, value)} />
                </div>
                <p className="text-xs text-muted-foreground">包含全部子任务的累计用量，达到上限后停止新增请求。</p>
              </>
            ) : null}
          </div>
        );
      })}
      <div className="flex flex-col gap-2">
        <label htmlFor={`${formId}-tasks`} className="text-sm font-medium">最多智能体任务</label>
        <CanvasNumberField id={`${formId}-tasks`} label="最多智能体任务" min={1} max={64} value={typeof config.maxAgentRuns === "number" ? config.maxAgentRuns : undefined}
          disabled={disabled} placeholder="默认 16" presets={[4, 8, 16, 32]} onChange={(value) => update("maxAgentRuns", value)} />
        <p className="text-xs text-muted-foreground">含本节点的主任务及所有层级的子任务。</p>
      </div>
        </div>
      </details>
    </section>
  );
}
