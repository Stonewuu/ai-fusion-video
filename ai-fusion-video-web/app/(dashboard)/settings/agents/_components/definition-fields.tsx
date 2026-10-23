"use client";

import { useId } from "react";
import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { AgentSkillPicker } from "@/components/agents/skill-picker";
import { effectiveAgentValues, type AgentDefinitionDetail, type AgentDefinitionOverrides, type AgentDefinitionValues } from "@/lib/api/agent-definitions";

export interface DefinitionFieldsProps {
  detail: AgentDefinitionDetail;
  draft: AgentDefinitionOverrides;
  onChange: (draft: AgentDefinitionOverrides) => void;
  disabled: boolean;
}

export function FieldHeading({ field, label, draft, onChange, disabled, htmlFor }: Pick<DefinitionFieldsProps, "draft" | "onChange" | "disabled"> & {
  field: keyof AgentDefinitionValues; label: string; htmlFor?: string;
}) {
  const customized = draft[field] != null;
  return <div className="flex flex-wrap items-center justify-between gap-2">
    <label htmlFor={htmlFor} className="text-sm font-medium">{label}</label>
    {customized ? <Button type="button" variant="ghost" size="xs" disabled={disabled} aria-label={`恢复内置${label}`} onClick={() => {
      const next = { ...draft }; delete next[field]; onChange(next);
    }}><RotateCcw />恢复内置</Button> : <span className="text-xs text-muted-foreground">使用内置设置</span>}
  </div>;
}

export function DefinitionPromptFields(props: DefinitionFieldsProps) {
  const id = useId();
  const { detail, draft, onChange, disabled } = props;
  const value = effectiveAgentValues(detail.configuration.defaults, draft);
  return <div className="flex flex-col gap-6">
    {([
      ["name", "显示名称", 100, ""],
      ["systemPrompt", "系统提示词", 131072, "定义智能体的职责、工作方式与约束。"],
      ["instructionTemplate", "任务指令模板", 65536, "在每次任务中补充具体要求。保留需要的 {变量名}；留空表示不追加。"],
      ["defaultUserMessage", "默认任务内容", 32768, "未填写任务要求时使用；留空表示不预设。"],
    ] as const).map(([field, label, limit, hint]) => <div key={field} className="flex flex-col gap-2">
      <FieldHeading {...props} field={field} label={label} htmlFor={`${id}-${field}`} />
      {field === "name" ? <Input id={`${id}-${field}`} required maxLength={limit} disabled={disabled} value={value[field]} onChange={(event) => onChange({ ...draft, [field]: event.target.value })} />
        : <Textarea id={`${id}-${field}`} required={field === "systemPrompt"} maxLength={limit} disabled={disabled} value={value[field]} className={field === "systemPrompt" ? "min-h-64 font-mono text-sm" : "min-h-24 font-mono text-sm"} onChange={(event) => onChange({ ...draft, [field]: event.target.value })} />}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>)}
  </div>;
}

export function DefinitionSkillFields(props: DefinitionFieldsProps) {
  const id = useId();
  const { detail, draft, onChange, disabled } = props;
  const value = effectiveAgentValues(detail.configuration.defaults, draft);
  return <div className="flex flex-col gap-6">
    <section className="flex flex-col gap-3">
      <FieldHeading {...props} field="defaultSkills" label="默认 Skill" />
      <p className="text-sm text-muted-foreground">用于新对话和使用此智能体的画布节点；节点可单独指定，也可不使用 Skill。</p>
      <AgentSkillPicker options={detail.skills} selected={value.defaultSkills} onChange={(defaultSkills) => onChange({ ...draft, defaultSkills })} disabled={disabled} />
    </section>
    <section className="flex flex-col gap-3 border-t border-border/20 pt-6">
      <FieldHeading {...props} field="maxIters" label="执行步数" htmlFor={`${id}-steps`} />
      <label className="flex items-center gap-3 text-sm">
        <Checkbox checked={value.maxIters > 0} disabled={disabled} onCheckedChange={(checked) => onChange({ ...draft, maxIters: checked ? 50 : 0 })} />限制执行步数
      </label>
      {value.maxIters > 0 ? <Input id={`${id}-steps`} type="number" min={1} max={2147483647} value={value.maxIters} disabled={disabled} onChange={(event) => onChange({ ...draft, maxIters: Number(event.target.value) })} />
        : <p className="text-sm text-muted-foreground">不限步数，持续执行到任务完成或主动停止。</p>}
    </section>
  </div>;
}
