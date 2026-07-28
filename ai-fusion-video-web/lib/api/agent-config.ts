import { http } from "./client";

export type AgentWorkspaceBackend = "database" | "local" | "object_storage";

export interface AgentWorkspaceMigration {
  id: number;
  sourceBackendType: AgentWorkspaceBackend;
  sourceStorageConfigId?: number | null;
  sourceLocalPath?: string | null;
  targetBackendType: AgentWorkspaceBackend;
  targetStorageConfigId?: number | null;
  targetLocalPath?: string | null;
  status: string;
  totalCount: number;
  copiedCount: number;
  failedCount: number;
  errorMessage?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
}

export interface AgentWorkspaceConfig {
  backendType: AgentWorkspaceBackend;
  storageConfigId?: number | null;
  localPath?: string | null;
  migrationStatus: string;
  activeMigrationId?: number | null;
  entryCount: number;
  contentBytes: number;
  latestMigration?: AgentWorkspaceMigration | null;
}

export interface AgentWorkspaceTarget {
  backendType: AgentWorkspaceBackend;
  storageConfigId?: number | null;
  localPath?: string | null;
}

export interface AgentUserSkill {
  id: string;
  name: string;
  displayName: string | null;
  description: string;
  content: string;
  source: string;
}

export interface AgentSkillSaveRequest {
  originalName?: string | null;
  name: string;
  displayName: string;
  description: string;
  content: string;
}

export interface AgentMcpServer {
  id: number;
  name: string;
  transport: "http" | "sse";
  url: string;
  headers: Record<string, string>;
  queryParams: Record<string, string>;
  enabledTools: string[];
  protocolVersions: string[];
  timeoutSeconds: number;
  initializationTimeoutSeconds: number;
  status: number;
  lastTestStatus?: string | null;
  lastTestMessage?: string | null;
  updateTime?: string | null;
}

export type AgentMcpServerSaveRequest = Omit<
  AgentMcpServer,
  "id" | "lastTestStatus" | "lastTestMessage" | "updateTime"
> & { id?: number };

export interface AgentMcpTestResult {
  success: boolean;
  message: string;
  tools: Array<{ name: string; description: string; readOnly: boolean }>;
}

export const agentConfigApi = {
  workspace(): Promise<AgentWorkspaceConfig> {
    return http.get("/api/ai/agent-config/workspace");
  },
  testWorkspace(target: AgentWorkspaceTarget): Promise<boolean> {
    return http.post("/api/ai/agent-config/workspace/test", target);
  },
  migrateWorkspace(target: AgentWorkspaceTarget): Promise<number> {
    return http.post("/api/ai/agent-config/workspace/migrations", target);
  },
  migration(id: number): Promise<AgentWorkspaceMigration> {
    return http.get(`/api/ai/agent-config/workspace/migrations/${id}`);
  },
  rollbackMigration(id: number): Promise<boolean> {
    return http.post(`/api/ai/agent-config/workspace/migrations/${id}/rollback`);
  },
  dismissMigrationFailure(id: number): Promise<boolean> {
    return http.post(`/api/ai/agent-config/workspace/migrations/${id}/dismiss-failure`);
  },
  skills(): Promise<AgentUserSkill[]> {
    return http.get("/api/ai/agent-config/skills");
  },
  saveSkill(request: AgentSkillSaveRequest): Promise<AgentUserSkill> {
    return http.put("/api/ai/agent-config/skills", request);
  },
  deleteSkill(name: string): Promise<boolean> {
    return http.delete(`/api/ai/agent-config/skills/${encodeURIComponent(name)}`);
  },
  mcpServers(): Promise<AgentMcpServer[]> {
    return http.get("/api/ai/agent-config/mcp");
  },
  saveMcpServer(request: AgentMcpServerSaveRequest): Promise<AgentMcpServer> {
    return http.put("/api/ai/agent-config/mcp", request);
  },
  deleteMcpServer(id: number): Promise<boolean> {
    return http.delete(`/api/ai/agent-config/mcp/${id}`);
  },
  testMcpServer(id: number): Promise<AgentMcpTestResult> {
    return http.post(`/api/ai/agent-config/mcp/${id}/test`);
  },
};
