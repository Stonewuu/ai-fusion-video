"use client";

import { useCallback, useEffect, useState } from "react";
import { Loader2, RotateCcw } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { agentDefinitionsApi, compactAgentOverrides, effectiveAgentValues, type AgentDefinitionDetail, type AgentDefinitionOverrides, type AgentDefinitionSummary } from "@/lib/api/agent-definitions";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { notifyAgentConfigurationChanged } from "@/lib/agent-configuration-events";
import { DefinitionPromptFields, DefinitionSkillFields } from "./definition-fields";
import { DefinitionToolFields } from "./definition-tools";

export function DefinitionEditor({ agent, userId, onClose, onSaved }: {
  agent: AgentDefinitionSummary;
  userId: number;
  onClose: () => void;
  onSaved: () => void;
}) {
  const { confirm } = useConfirm();
  const [detail, setDetail] = useState<AgentDefinitionDetail | null>(null);
  const [draft, setDraft] = useState<AgentDefinitionOverrides>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conflict, setConflict] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const dirty = !!detail && JSON.stringify(compactAgentOverrides(draft)) !== JSON.stringify(compactAgentOverrides(detail.configuration.overrides));
  useEffect(() => {
    let active = true;
    void agentDefinitionsApi.get(agent.type).then((value) => {
      if (!active) return;
      setDetail(value); setDraft(compactAgentOverrides(value.configuration.overrides));
    }).catch((failure) => { if (active) setError(getApiErrorMessage(failure)); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [agent.type, userId, attempt]);
  useEffect(() => {
    if (!dirty) return;
    const guard = (event: BeforeUnloadEvent) => { event.preventDefault(); };
    window.addEventListener("beforeunload", guard);
    return () => window.removeEventListener("beforeunload", guard);
  }, [dirty]);

  const close = useCallback(async () => {
    if (saving) return;
    if (dirty && !await confirm({ title: "放弃未保存的修改？", description: "当前修改尚未应用到智能体。", confirmText: "放弃修改", cancelText: "继续编辑", variant: "warning" })) return;
    onClose();
  }, [confirm, dirty, onClose, saving]);
  const reload = async () => {
    if (dirty && !await confirm({ title: "重新载入最新配置？", description: "将用已保存的配置替换当前未保存的修改。", confirmText: "重新载入", cancelText: "保留修改", variant: "warning" })) return;
    setLoading(true); setError(null); setConflict(false); setAttempt((value) => value + 1);
  };
  const save = async () => {
    if (!detail) return;
    const value = effectiveAgentValues(detail.configuration.defaults, draft);
    if (!value.name.trim() || !value.systemPrompt.trim()) { setError("请填写显示名称和系统提示词。"); return; }
    if (!Number.isSafeInteger(value.maxIters) || value.maxIters < 0 || value.maxIters > 2147483647) { setError("执行步数需为有效整数，0 表示不限步数。"); return; }
    setSaving(true); setError(null);
    try {
      const saved = await agentDefinitionsApi.save(agent.type, detail.configuration.revision, compactAgentOverrides(draft));
      setDetail(saved); setDraft(compactAgentOverrides(saved.configuration.overrides));
      notifyAgentConfigurationChanged(userId);
      toast.success("智能体配置已保存，新任务将使用此配置");
      onSaved(); onClose();
    } catch (failure) {
      setError(getApiErrorMessage(failure)); setConflict(getApiErrorStatus(failure) === 409);
    } finally { setSaving(false); }
  };
  const fields = detail ? { detail, draft, onChange: setDraft, disabled: saving } : null;
  return <Dialog open onOpenChange={(open) => { if (!open) void close(); }}>
    <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-3xl">
      <DialogHeader className="shrink-0 pr-8">
        <DialogTitle>配置 {agent.builtinName}</DialogTitle>
        <DialogDescription>仅对你的账号生效。保存后用于新任务，正在运行的任务保持原配置。</DialogDescription>
      </DialogHeader>
      {loading ? <div role="status" className="flex min-h-48 items-center justify-center gap-2 text-muted-foreground"><Loader2 className="size-4 animate-spin" />加载配置中…</div>
        : fields && <Tabs defaultValue="prompts" className="min-h-0 gap-4 overflow-hidden">
          <div className="shrink-0 overflow-x-auto"><TabsList aria-label="智能体配置内容">
            <TabsTrigger value="prompts" className="transition-colors motion-reduce:transition-none">提示词</TabsTrigger>
            <TabsTrigger value="tools" className="transition-colors motion-reduce:transition-none">工具</TabsTrigger>
            <TabsTrigger value="skills" className="transition-colors motion-reduce:transition-none">Skill 与执行</TabsTrigger>
          </TabsList></div>
          <div className="min-h-0 overflow-y-auto px-1 py-1">
            <TabsContent value="prompts"><DefinitionPromptFields {...fields} /></TabsContent>
            <TabsContent value="tools"><DefinitionToolFields {...fields} /></TabsContent>
            <TabsContent value="skills"><DefinitionSkillFields {...fields} /></TabsContent>
          </div>
        </Tabs>}
      {error && <div role="alert" className="shrink-0 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"><p>{error}</p>
        {(conflict || !detail) && <Button variant="outline" size="sm" disabled={saving || loading} onClick={() => void reload()}>{conflict ? "载入最新配置" : "重试"}</Button>}
      </div>}
      <DialogFooter className="shrink-0 border-t border-border/20 pt-4">
        <div className="mr-auto flex items-center gap-2">
          <Button variant="ghost" disabled={saving || !detail || !Object.keys(compactAgentOverrides(draft)).length} onClick={() => setDraft({})}><RotateCcw />全部恢复内置</Button>
          {dirty && <span className="text-xs text-muted-foreground" role="status">未保存</span>}
        </div>
        <Button variant="outline" disabled={saving} onClick={() => void close()}>取消</Button>
        <Button disabled={saving || loading || !dirty || conflict} onClick={() => void save()}>{saving && <Loader2 className="animate-spin" />}保存配置</Button>
      </DialogFooter>
    </DialogContent>
  </Dialog>;
}
