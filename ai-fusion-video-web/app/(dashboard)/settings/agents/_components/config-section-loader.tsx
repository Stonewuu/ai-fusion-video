"use client";

import { useCallback, useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { agentConfigApi } from "@/lib/api/agent-config";
import { storageConfigApi } from "@/lib/api/storage";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { notifyAgentConfigurationChanged } from "@/lib/agent-configuration-events";
import { SkillsSection } from "./skills-section";
import { McpServersSection } from "./mcp-servers-section";
import { StateRetentionSection } from "./state-retention-section";
import { WorkspaceStorageSection } from "./workspace-storage-section";

async function fetchSection(section: string, isAdmin: boolean) {
  if (section === "skills") return { kind: "skills" as const, skills: await agentConfigApi.skills() };
  if (section === "mcp") return { kind: "mcp" as const, servers: await agentConfigApi.mcpServers() };
  const [workspace, policy, stores] = await Promise.all([
    agentConfigApi.workspace(), agentConfigApi.stateCleanupPolicy(), isAdmin ? storageConfigApi.list() : Promise.resolve([]),
  ]);
  return { kind: "runtime" as const, workspace, policy, stores };
}

export function ConfigSectionLoader({ section, isAdmin, userId }: { section: string; isAdmin: boolean; userId: number }) {
  const [data, setData] = useState<Awaited<ReturnType<typeof fetchSection>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    void fetchSection(section, isAdmin).then((value) => { if (active) { setData(value); setError(null); } })
      .catch((failure) => { if (active) setError(getApiErrorMessage(failure)); });
    return () => { active = false; };
  }, [section, isAdmin, userId, attempt]);
  const refresh = useCallback(async () => {
    // 保存与刷新分开反馈：刷新失败不把已经成功的保存误报为失败。
    notifyAgentConfigurationChanged(userId);
    try { setData(await fetchSection(section, isAdmin)); setError(null); }
    catch (failure) { setError(getApiErrorMessage(failure)); }
  }, [section, isAdmin, userId]);
  const migrationActive = data?.kind === "runtime" && ["copying", "verifying", "cutover"].includes(data.workspace.migrationStatus);
  useEffect(() => {
    if (!migrationActive) return;
    let active = true;
    const timer = window.setInterval(() => {
      void agentConfigApi.workspace().then((workspace) => {
        if (active) setData((value) => value?.kind === "runtime" ? { ...value, workspace } : value);
      }).catch((failure) => { if (active) setError(getApiErrorMessage(failure)); });
    }, 2000);
    return () => { active = false; window.clearInterval(timer); };
  }, [migrationActive]);
  return <div className="space-y-6">
    {error && <div role="alert" className="flex items-center gap-3 rounded-xl border border-destructive/30 p-4"><p className="flex-1 text-sm">{error}</p><Button variant="outline" onClick={() => { setError(null); setAttempt((value) => value + 1); }}>重试</Button></div>}
    {!data && !error && <div role="status" className="flex min-h-48 items-center justify-center gap-2 text-muted-foreground"><Loader2 className="size-4 animate-spin" />加载配置中…</div>}
    {data?.kind === "skills" && <SkillsSection skills={data.skills} onRefresh={refresh} />}
    {data?.kind === "mcp" && <McpServersSection servers={data.servers} onRefresh={refresh} />}
    {data?.kind === "runtime" && <>
      <StateRetentionSection policy={data.policy} isAdmin={isAdmin} onSaved={(policy) => setData({ ...data, policy })} />
      <WorkspaceStorageSection config={data.workspace} storageConfigs={data.stores} isAdmin={isAdmin} onRefresh={refresh} />
    </>}
  </div>;
}
