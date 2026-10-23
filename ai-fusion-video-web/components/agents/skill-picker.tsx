"use client";

import { useId, useState } from "react";
import { Search } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import type { AgentSkillOption } from "@/lib/api/agent-definitions";
import { cn } from "@/lib/utils";

export function AgentSkillPicker({ options, selected, onChange, disabled = false }: {
  options: AgentSkillOption[];
  selected: string[];
  onChange: (value: string[]) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const [query, setQuery] = useState("");
  const unavailable = selected.filter((name) => !options.some((option) => option.name === name));
  const visible = options.filter((option) => `${option.name} ${option.displayName} ${option.description}`.toLowerCase().includes(query.trim().toLowerCase()));
  const toggle = (name: string, checked: boolean) => onChange(checked ? [...selected, name] : selected.filter((value) => value !== name));
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>按选择顺序加载工作规范</span><span aria-live="polite">已选 {selected.length} / 8</span>
      </div>
      {options.length > 5 && <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-3 size-4 text-muted-foreground" />
        <Input aria-label="搜索 Skill" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索名称或用途" className="pl-9" />
      </div>}
      <div className="divide-y divide-border/20 rounded-lg border border-border/20 bg-background/70">
        {unavailable.map((name, index) => (
          <label key={name} htmlFor={`${id}-missing-${index}`} className="flex cursor-pointer items-start gap-3 p-3 text-sm text-destructive">
            <Checkbox id={`${id}-missing-${index}`} checked disabled={disabled} onCheckedChange={() => toggle(name, false)} />
            <span className="min-w-0 break-words">{name} <span className="text-xs">已不可用，取消勾选以移除</span></span>
          </label>
        ))}
        {visible.map((option, index) => {
          const checked = selected.includes(option.name);
          const locked = disabled || !checked && selected.length >= 8;
          return <label key={option.name} htmlFor={`${id}-${index}`} className={cn("flex items-start gap-3 p-3 transition-colors motion-reduce:transition-none", locked ? "opacity-60" : "cursor-pointer hover:bg-muted/40", checked && "bg-primary/5")}>
            <Checkbox id={`${id}-${index}`} aria-label={option.displayName ?? option.name} checked={checked} disabled={locked} onCheckedChange={(value) => toggle(option.name, value)} />
            <span className="flex min-w-0 flex-1 flex-col gap-1">
              <span className="break-words text-sm font-medium">{option.displayName ?? option.name}</span>
              <span className="break-words text-xs text-muted-foreground">{option.description}</span>
              <span className="break-all text-xs text-muted-foreground">{option.name}</span>
            </span>
          </label>;
        })}
        {!visible.length && <p className="p-4 text-sm text-muted-foreground">{query ? "没有匹配的 Skill" : "还没有可选的 Skill，可在智能体配置中创建或导入。"}</p>}
      </div>
    </div>
  );
}
