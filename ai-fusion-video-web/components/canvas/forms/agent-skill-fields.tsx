"use client";

import { ExternalLink } from "lucide-react";
import { AgentSkillPicker } from "@/components/agents/skill-picker";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { CanvasAgentProfile, CanvasConfig } from "@/lib/canvas/types";
import { useCanvas } from "../canvas-provider";
import { CanvasSegmentedControl } from "../workspace/segmented-control";

export function CanvasAgentSkillFields({ profile, config, onChange, disabled }: {
  profile: CanvasAgentProfile | undefined;
  config: CanvasConfig;
  onChange: (value: CanvasConfig) => void;
  disabled: boolean;
}) {
  const skills = useCanvas((state) => state.catalog!.skills);
  const custom = config.skillMode === "CUSTOM";
  const names = custom ? Array.isArray(config.skillNames) ? config.skillNames : [] : profile?.defaultSkills ?? [];
  return <section className="flex flex-col gap-3 border-t border-border/20 pt-4" aria-label="节点 Skill">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h3 className="text-sm font-medium">Skill 工作规范</h3>
      <Button variant="ghost" size="xs" nativeButton={false} render={<a href="/settings/agents?section=skills" target="_blank" rel="noopener noreferrer" />} title="在新标签页管理 Skill，保留当前画布">管理<ExternalLink /></Button>
    </div>
    <CanvasSegmentedControl label="Skill 使用方式" value={custom ? "CUSTOM" : "INHERIT"}
      options={[{ value: "INHERIT", label: "跟随智能体", disabled }, { value: "CUSTOM", label: "单独指定", disabled }]}
      onChange={(mode) => {
        const next: CanvasConfig = { ...config, skillMode: mode };
        if (mode === "INHERIT") delete next.skillNames;
        else next.skillNames = [...names];
        onChange(next);
      }} />
    {custom ? <>
      <AgentSkillPicker options={skills} selected={names} onChange={(skillNames) => onChange({ ...config, skillMode: "CUSTOM", skillNames })} disabled={disabled} />
      <p className="text-xs text-muted-foreground">用于本节点及其子任务；不选择表示不使用 Skill。</p>
    </> : <>
      {names.length ? <div className="flex flex-wrap gap-2">{names.map((name) => <Badge key={name} variant="secondary">{skills.find((skill) => skill.name === name)?.displayName ?? name}</Badge>)}</div>
        : <p className="text-xs text-muted-foreground">此智能体未预设 Skill。可单独选择本次需要的工作规范。</p>}
      {!!names.length && <p className="text-xs text-muted-foreground">子任务也会使用各自的默认 Skill。</p>}
    </>}
  </section>;
}
