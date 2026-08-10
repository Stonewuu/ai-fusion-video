"use client";

import { useCallback, useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { storageConfigApi, type StorageConfig } from "@/lib/api/storage";
import {
  agentConfigApi,
  type AgentMcpServer,
  type AgentUserSkill,
  type AgentWorkspaceConfig,
  type AgentStateCleanupPolicy,
} from "@/lib/api/agent-config";
import { toastApiError } from "@/lib/api/toast-api-error";
import { useAuthStore } from "@/lib/store/auth-store";
import { McpServersSection } from "./_components/mcp-servers-section";
import { SkillsSection } from "./_components/skills-section";
import { WorkspaceStorageSection } from "./_components/workspace-storage-section";
import { StateRetentionSection } from "./_components/state-retention-section";
import { settingsTypography } from "../_shared";

const WORKSPACE_LABELS = {
  database: "数据库",
  local: "本地存储",
  object_storage: "对象存储",
} as const;

export default function AgentSettingsPage() {
  const currentUser = useAuthStore((state) => state.user);
  const isAdmin = currentUser?.roles?.includes("admin") ?? false;
  const [workspace, setWorkspace] = useState<AgentWorkspaceConfig | null>(null);
  const [statePolicy, setStatePolicy] = useState<AgentStateCleanupPolicy | null>(null);
  const [skills, setSkills] = useState<AgentUserSkill[]>([]);
  const [mcpServers, setMcpServers] = useState<AgentMcpServer[]>([]);
  const [storageConfigs, setStorageConfigs] = useState<StorageConfig[]>([]);
  const [loading, setLoading] = useState(true);

  const loadWorkspace = useCallback(async () => {
    setWorkspace(await agentConfigApi.workspace());
  }, []);

  const loadAll = useCallback(async () => {
    const [workspaceConfig, cleanupPolicy, skillList, mcpList, storageList] = await Promise.all([
      agentConfigApi.workspace(),
      agentConfigApi.stateCleanupPolicy(),
      agentConfigApi.skills(),
      agentConfigApi.mcpServers(),
      isAdmin ? storageConfigApi.list() : Promise.resolve([]),
    ]);
    setWorkspace(workspaceConfig);
    setStatePolicy(cleanupPolicy);
    setSkills(skillList);
    setMcpServers(mcpList);
    setStorageConfigs(storageList);
  }, [isAdmin]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(() => {
      loadAll()
        .catch((error) => {
          if (!cancelled) toast.error(error instanceof Error ? error.message : "加载智能体配置失败");
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [loadAll]);

  useEffect(() => {
    if (!workspace || !["copying", "verifying", "cutover"].includes(workspace.migrationStatus)) return;
    const timer = window.setInterval(() => {
      loadWorkspace().catch((error) => {
        toastApiError(error, "刷新工作空间迁移状态失败");
      });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [loadWorkspace, workspace]);

  const refreshSkills = useCallback(async () => setSkills(await agentConfigApi.skills()), []);
  const refreshMcp = useCallback(async () => setMcpServers(await agentConfigApi.mcpServers()), []);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className={settingsTypography.pageTitle}>智能体配置</h1>
          <p className={settingsTypography.pageDescription}>
            管理 AgentScope 工作空间、状态生命周期，以及助手可主动引用的 Skill 和 MCP 工具。
          </p>
        </div>
        {workspace && (
          <div className="flex flex-wrap items-center gap-2" aria-label="智能体配置摘要">
            <Badge variant="outline">{WORKSPACE_LABELS[workspace.backendType]}</Badge>
            <Badge variant="secondary">{skills.length} 个 Skill</Badge>
            <Badge variant="secondary">{mcpServers.length} 个 MCP</Badge>
          </div>
        )}
      </header>

      {loading || !workspace || !statePolicy ? (
        <div className="flex min-h-64 items-center justify-center text-muted-foreground">
          <Loader2 className="mr-2 size-5 animate-spin" />加载配置中…
        </div>
      ) : (
        <div className="space-y-6">
          <WorkspaceStorageSection
            config={workspace}
            storageConfigs={storageConfigs}
            isAdmin={isAdmin}
            onRefresh={loadWorkspace}
          />
          <StateRetentionSection
            policy={statePolicy}
            isAdmin={isAdmin}
            onSaved={setStatePolicy}
          />
          <div className="grid items-start gap-6 xl:grid-cols-2">
            <SkillsSection skills={skills} onRefresh={refreshSkills} />
            <McpServersSection servers={mcpServers} onRefresh={refreshMcp} />
          </div>
        </div>
      )}
    </div>
  );
}
