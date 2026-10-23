"use client";

import { useEffect, useState } from "react";
import { Bot, Loader2, Search, SlidersHorizontal } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { agentDefinitionsApi, type AgentDefinitionSummary } from "@/lib/api/agent-definitions";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { DefinitionEditor } from "./definition-editor";

export function DefinitionsSection({ userId }: { userId: number }) {
  const [agents, setAgents] = useState<AgentDefinitionSummary[]>([]);
  const [editing, setEditing] = useState<AgentDefinitionSummary | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let active = true;
    void agentDefinitionsApi.list().then((result) => { if (active) { setAgents(result); setError(null); } })
      .catch((failure) => { if (active) setError(getApiErrorMessage(failure)); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [userId, revision]);
  const filtered = agents.filter((agent) => `${agent.name} ${agent.builtinName} ${agent.type}`.toLowerCase().includes(query.trim().toLowerCase()));
  return <section className="flex flex-col gap-4" aria-label="智能体列表">
    <div className="flex flex-wrap items-center justify-between gap-4">
      <div><h2 className="text-base font-semibold">我的智能体配置</h2><p className="mt-1 text-sm text-muted-foreground">从内置能力开始，调整提示词、工具和默认 Skill。</p></div>
      <div className="relative w-full sm:w-72"><Search className="pointer-events-none absolute left-3 top-3 size-4 text-muted-foreground" /><Input aria-label="搜索智能体" placeholder="搜索智能体" value={query} onChange={(event) => setQuery(event.target.value)} className="pl-9" /></div>
    </div>
    {loading ? <div role="status" className="flex min-h-48 items-center justify-center gap-2 text-muted-foreground"><Loader2 className="size-4 animate-spin" />加载智能体中…</div>
      : error ? <div role="alert" className="flex items-center gap-3 rounded-xl border border-destructive/30 p-4"><span className="text-sm">{error}</span><Button variant="outline" onClick={() => { setLoading(true); setRevision((value) => value + 1); }}>重试</Button></div>
        : <div className="divide-y divide-border/20 rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm">
          {filtered.map((agent) => <div key={agent.type} className="flex items-center gap-3 p-4">
            <Bot className="size-5 shrink-0 text-primary" />
            <div className="flex min-w-0 flex-1 flex-col gap-1">
              <div className="flex flex-wrap items-center gap-2"><h3 className="text-sm font-medium">{agent.name}</h3><Badge variant={agent.customized ? "secondary" : "outline"}>{agent.customized ? "已自定义" : "内置设置"}</Badge></div>
              <p className="text-xs text-muted-foreground">{agent.name !== agent.builtinName && `${agent.builtinName} · `}{agent.toolsEnabled ? "可使用工具" : "仅生成文字"} · {agent.defaultSkills.length ? `${agent.defaultSkills.length} 个默认 Skill` : "未预设 Skill"} · {agent.maxIters ? `最多 ${agent.maxIters} 步` : "不限步数"}</p>
            </div>
            <Button variant="outline" onClick={() => setEditing(agent)} aria-label={`配置 ${agent.name}`}><SlidersHorizontal />配置</Button>
          </div>)}
          {!filtered.length && <p className="p-6 text-sm text-muted-foreground">没有匹配的智能体，试试其他名称。</p>}
        </div>}
    {editing && <DefinitionEditor key={editing.type} agent={editing} userId={userId} onClose={() => setEditing(null)} onSaved={() => setRevision((value) => value + 1)} />}
  </section>;
}
