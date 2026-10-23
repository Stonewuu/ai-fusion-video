"use client";

import { useId, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectGroup, SelectItem } from "@/components/ui/select";
import { effectiveAgentValues, type AgentToolOption } from "@/lib/api/agent-definitions";
import { FieldHeading, type DefinitionFieldsProps } from "./definition-fields";

const categories = ["项目管理", "剧本创作", "分镜创作", "资产管理", "媒体生成", "画布操作", "子任务委派", "MCP 外部服务"];
const toolGroups = [
  ["platform", "toolNames", "平台工具"], ["agent", "subAgentToolNames", "子任务工具"], ["mcp", "mcpToolNames", "MCP 工具"],
] as const;

function ToolChoice({ tool, checked, disabled, onChange }: {
  tool: AgentToolOption; checked: boolean; disabled: boolean; onChange: (checked: boolean) => void;
}) {
  const id = useId();
  const [expanded, setExpanded] = useState(false);
  const longDescription = tool.description.length > 120;
  return <div className="p-3">
    <div className="flex items-start gap-3">
      <Checkbox id={id} aria-label={tool.label} checked={checked} disabled={disabled} onCheckedChange={onChange} />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <label htmlFor={id} className="cursor-pointer text-sm font-medium">{tool.label} <span className="text-xs font-normal text-muted-foreground">{tool.readOnly ? "读取" : "执行"}</span></label>
        <p className={`${expanded ? "whitespace-pre-wrap" : "line-clamp-2"} break-words text-xs text-muted-foreground`}>{tool.description}</p>
        {longDescription && <Button variant="link" size="xs" className="self-start" aria-expanded={expanded} aria-label={`${expanded ? "收起" : "查看"}${tool.label}的完整说明`} onClick={() => setExpanded(!expanded)}>{expanded ? "收起说明" : "完整说明"}</Button>}
      </div>
    </div>
  </div>;
}

export function DefinitionToolFields(props: DefinitionFieldsProps) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [selectedOnly, setSelectedOnly] = useState(false);
  const { detail, draft, onChange, disabled } = props;
  const value = effectiveAgentValues(detail.configuration.defaults, draft);
  const items = [{ value: "all", label: "全部分类" }, ...categories.filter((name) => detail.tools.some((tool) => tool.category === name)).map((name) => ({ value: name, label: name }))];
  const search = query.trim().toLowerCase();
  return <div className="flex flex-col gap-6">
    <div className="flex flex-col gap-3">
      <FieldHeading {...props} field="toolsEnabled" label="工具调用" htmlFor={`${id}-enabled`} />
      <label htmlFor={`${id}-enabled`} className="flex items-center gap-3 text-sm">
        <Checkbox id={`${id}-enabled`} checked={value.toolsEnabled} disabled={disabled} onCheckedChange={(checked) => onChange({ ...draft, toolsEnabled: checked })} />
        允许智能体使用工具与委派子任务
      </label>
      {!value.toolsEnabled && <p className="text-sm text-muted-foreground">当前只生成文字。以下选择会保留，重新开启后生效。</p>}
    </div>
    <div className="flex flex-col gap-3">
      <Input aria-label="搜索工具" placeholder="跨分类搜索名称、标识或用途" value={query} onChange={(event) => setQuery(event.target.value)} />
      <div className="flex flex-wrap items-center gap-3">
        <Select items={items} value={category} onValueChange={(next) => setCategory(String(next))}>
          <SelectTrigger aria-label="工具分类" className="w-44 text-sm"><SelectValue placeholder="选择工具分类" /></SelectTrigger>
          <SelectContent className="text-sm"><SelectGroup>{items.map((item) => <SelectItem className="text-sm" key={item.value} value={item.value}>{item.label}</SelectItem>)}</SelectGroup></SelectContent>
        </Select>
        <label className="flex items-center gap-2 text-sm"><Checkbox checked={selectedOnly} onCheckedChange={setSelectedOnly} />只看已选</label>
        {(search || category !== "all" || selectedOnly) && <Button variant="ghost" size="sm" onClick={() => { setQuery(""); setCategory("all"); setSelectedOnly(false); }}>清除筛选</Button>}
      </div>
    </div>
    {toolGroups.map(([kind, field, label]) => {
      const options = detail.tools.filter((tool) => tool.kind === kind);
      const selected = value[field] ?? options.map((tool) => tool.name);
      const missing = selected.filter((name) => !options.some((tool) => tool.name === name));
      const visible = options.filter((tool) => (category === "all" || category === tool.category) && (!selectedOnly || selected.includes(tool.name))
        && `${tool.label} ${tool.name} ${tool.description}`.toLowerCase().includes(search));
      if (!visible.length && !missing.length && (search || category !== "all" || selectedOnly)) return null;
      return <section key={kind} className="flex flex-col gap-3" aria-label={label}>
        <FieldHeading {...props} field={field} label={label} />
        <p className="text-xs text-muted-foreground">已选 {selected.length} / {options.length} · 当前显示 {visible.length} 项</p>
        {missing.map((name) => <label key={name} className="flex items-start gap-3 text-sm text-destructive">
          <Checkbox aria-label={`移除不可用工具 ${name}`} checked disabled={disabled} onCheckedChange={() => onChange({ ...draft, [field]: selected.filter((item) => item !== name) })} />{name} 已不可用，取消勾选以移除
        </label>)}
        {categories.map((group) => {
          const shown = visible.filter((tool) => tool.category === group);
          if (!shown.length) return null;
          return <section key={group} aria-label={group} className="overflow-hidden rounded-lg border border-border/20 bg-background/70">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/20 bg-muted/30 p-3">
              <h4 className="text-sm font-medium">{group} <span className="font-normal text-muted-foreground">{shown.filter((tool) => selected.includes(tool.name)).length} / {shown.length}</span></h4>
              <div className="flex gap-2">
                <Button variant="ghost" size="xs" aria-label={`全选当前${group}`} disabled={disabled || !value.toolsEnabled || shown.every((tool) => selected.includes(tool.name))} onClick={() => onChange({ ...draft, [field]: [...new Set([...selected, ...shown.map((tool) => tool.name)])] })}>全选</Button>
                <Button variant="ghost" size="xs" aria-label={`清空当前${group}`} disabled={disabled || !value.toolsEnabled || !shown.some((tool) => selected.includes(tool.name))} onClick={() => onChange({ ...draft, [field]: selected.filter((name) => !shown.some((tool) => tool.name === name)) })}>清空</Button>
              </div>
            </div>
            <div className="divide-y divide-border/20">{shown.map((tool) => <ToolChoice key={tool.name} tool={tool} checked={selected.includes(tool.name)} disabled={disabled || !value.toolsEnabled} onChange={(checked) => onChange({ ...draft, [field]: checked ? [...selected, tool.name] : selected.filter((name) => name !== tool.name) })} />)}</div>
          </section>;
        })}
        {!options.length && <p className="text-sm text-muted-foreground">{kind === "agent" ? "此智能体没有预设子任务工具" : "还没有可用的工具"}</p>}
      </section>;
    })}
    {!detail.tools.some((tool) => {
      const field = toolGroups.find(([kind]) => kind === tool.kind)![1];
      return (category === "all" || category === tool.category) && (!selectedOnly || value[field] === null || value[field].includes(tool.name))
        && `${tool.label} ${tool.name} ${tool.description}`.toLowerCase().includes(search);
    }) && (search || selectedOnly || category !== "all") && <p role="status" className="text-sm text-muted-foreground">没有匹配的工具，可更换关键词或清除筛选。</p>}
  </div>;
}
