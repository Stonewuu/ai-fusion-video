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

export interface AgentStateCleanupPolicy {
  cleanupIntervalDays: number;
  retentionDays: number;
  nextCleanupAt: string;
  lastCleanupAt?: string | null;
}

export interface AgentStateCleanupPolicySaveRequest {
  cleanupIntervalDays: number;
  retentionDays: number;
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

export type AgentSkillImportAction = "CREATE" | "REPLACE" | "SKIP";

export interface AgentSkillImportCandidate {
  rootPath: string;
  name: string | null;
  description: string | null;
  suggestedDisplayName: string;
  fileCount: number;
  totalBytes: number;
  hasScripts: boolean;
  hasReferences: boolean;
  hasAssets: boolean;
  exists: boolean;
  valid: boolean;
  recommendedAction: AgentSkillImportAction;
  warnings: string[];
  errors: string[];
}

export interface AgentSkillImportPreview {
  sourceType: "ZIP" | "DIRECTORY";
  totalFiles: number;
  totalBytes: number;
  skills: AgentSkillImportCandidate[];
  warnings: string[];
  errors: string[];
}

export interface AgentSkillImportSelection {
  rootPath: string;
  displayName: string;
  action: AgentSkillImportAction;
}

export interface AgentSkillImportResult {
  createdCount: number;
  replacedCount: number;
  skippedCount: number;
  createdNames: string[];
  replacedNames: string[];
  skippedNames: string[];
}

function skillImportForm(files: File[], paths: string[], request?: object) {
  const form = new FormData();
  files.forEach((file, index) => {
    form.append("files", file, file.name);
    form.append("paths", paths[index] ?? file.name);
  });
  if (request) {
    form.append(
      "request",
      new Blob([JSON.stringify(request)], { type: "application/json" }),
    );
  }
  return form;
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
  stateCleanupPolicy(): Promise<AgentStateCleanupPolicy> {
    return http.get("/api/ai/agent-config/state-cleanup");
  },
  saveStateCleanupPolicy(
    request: AgentStateCleanupPolicySaveRequest,
  ): Promise<AgentStateCleanupPolicy> {
    return http.put("/api/ai/agent-config/state-cleanup", request);
  },
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
  previewSkillImport(files: File[], paths: string[]): Promise<AgentSkillImportPreview> {
    return http.post(
      "/api/ai/agent-config/skills/import/preview",
      skillImportForm(files, paths),
      { timeout: 120000 },
    );
  },
  importSkills(
    files: File[],
    paths: string[],
    selections: AgentSkillImportSelection[],
  ): Promise<AgentSkillImportResult> {
    return http.post(
      "/api/ai/agent-config/skills/import",
      skillImportForm(files, paths, { selections }),
      { timeout: 120000 },
    );
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
