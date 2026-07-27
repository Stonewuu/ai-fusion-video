"use client";

import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  AlertCircle,
  CheckCircle2,
  Cloud,
  Database,
  Folder,
  HardDrive,
  Loader2,
  RotateCcw,
  ServerCog,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  agentConfigApi,
  type AgentWorkspaceBackend,
  type AgentWorkspaceConfig,
  type AgentWorkspaceTarget,
} from "@/lib/api/agent-config";
import type { StorageConfig } from "@/lib/api/storage";

const BACKEND_ITEMS = [
  { value: "database", label: "数据库" },
  { value: "local", label: "本地存储" },
  { value: "object_storage", label: "对象存储" },
];

const BACKEND_LABELS: Record<AgentWorkspaceBackend, string> = {
  database: "数据库",
  local: "本地存储",
  object_storage: "对象存储",
};

const MIGRATION_STATUS_LABELS: Record<string, string> = {
  idle: "运行正常",
  copying: "正在复制",
  verifying: "正在校验",
  cutover: "正在切换",
  failed: "迁移失败",
};

interface WorkspaceStorageSectionProps {
  config: AgentWorkspaceConfig;
  storageConfigs: StorageConfig[];
  isAdmin: boolean;
  onRefresh: () => Promise<void>;
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

export function WorkspaceStorageSection({
  config,
  storageConfigs,
  isAdmin,
  onRefresh,
}: WorkspaceStorageSectionProps) {
  const [backendType, setBackendType] = useState<AgentWorkspaceBackend>(config.backendType);
  const [storageConfigId, setStorageConfigId] = useState<string>(
    config.storageConfigId ? String(config.storageConfigId) : ""
  );
  const [localPath, setLocalPath] = useState(config.localPath || "");
  const [testing, setTesting] = useState(false);
  const [migrating, setMigrating] = useState(false);
  const [rollingBack, setRollingBack] = useState(false);
  const [dismissing, setDismissing] = useState(false);

  useEffect(() => {
    setBackendType(config.backendType);
    setStorageConfigId(config.storageConfigId ? String(config.storageConfigId) : "");
    setLocalPath(config.localPath || "");
  }, [config.backendType, config.localPath, config.storageConfigId]);

  const objectStorageItems = useMemo(
    () => storageConfigs
      .filter((item) => item.type !== "local" && item.status === 1)
      .map((item) => ({ value: String(item.id), label: item.name })),
    [storageConfigs]
  );
  const currentStorageName = useMemo(() => {
    if (config.backendType !== "object_storage" || !config.storageConfigId) return null;
    return storageConfigs.find((item) => item.id === config.storageConfigId)?.name || `配置 #${config.storageConfigId}`;
  }, [config.backendType, config.storageConfigId, storageConfigs]);
  const target: AgentWorkspaceTarget = {
    backendType,
    storageConfigId: backendType === "object_storage" && storageConfigId
      ? Number(storageConfigId)
      : null,
    localPath: backendType === "local" ? localPath : null,
  };
  const activeMigration = config.latestMigration && [
    "copying",
    "verifying",
    "cutover",
  ].includes(config.latestMigration.status)
    ? config.latestMigration
    : null;
  const progress = activeMigration && activeMigration.totalCount > 0
    ? Math.min(100, Math.round(activeMigration.copiedCount / activeMigration.totalCount * 100))
    : activeMigration ? 5 : 0;
  const canRollback = config.latestMigration?.status === "completed";
  const targetMatchesCurrent = backendType === config.backendType
    && (backendType === "database"
      || (backendType === "local" && (localPath || "") === (config.localPath || ""))
      || (backendType === "object_storage" && Number(storageConfigId) === config.storageConfigId));
  const targetReady = backendType !== "object_storage" || Boolean(storageConfigId);
  const controlsDisabled = !isAdmin || config.migrationStatus !== "idle";

  const testTarget = async () => {
    setTesting(true);
    try {
      await agentConfigApi.testWorkspace(target);
      toast.success("目标存储连接与读写校验通过");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "目标存储测试失败");
    } finally {
      setTesting(false);
    }
  };

  const migrate = async () => {
    setMigrating(true);
    try {
      await agentConfigApi.migrateWorkspace(target);
      toast.success("工作空间迁移已开始");
      await onRefresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "启动迁移失败");
    } finally {
      setMigrating(false);
    }
  };

  const rollback = async () => {
    if (!config.latestMigration) return;
    setRollingBack(true);
    try {
      await agentConfigApi.rollbackMigration(config.latestMigration.id);
      toast.success("工作空间已回滚到原存储");
      await onRefresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "回滚失败");
    } finally {
      setRollingBack(false);
    }
  };

  const dismissFailure = async () => {
    if (!config.latestMigration) return;
    setDismissing(true);
    try {
      await agentConfigApi.dismissMigrationFailure(config.latestMigration.id);
      toast.success("失败状态已解除，可以重新测试或迁移");
      await onRefresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "解除失败状态失败");
    } finally {
      setDismissing(false);
    }
  };

  return (
    <section className="rounded-xl border border-border/30 bg-card/50 p-4 backdrop-blur-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="rounded-xl bg-primary/10 p-2 text-primary">
            <ServerCog className="size-5" />
          </div>
          <div>
            <h2 className="font-medium">工作空间存储</h2>
            <p className="mt-1 text-sm text-muted-foreground">选择智能体工作文件的持久化位置，并在不同后端之间安全迁移。</p>
          </div>
        </div>
        <Badge variant={config.migrationStatus === "failed" ? "destructive" : "outline"}>
          {config.migrationStatus === "idle" && <CheckCircle2 />}
          {MIGRATION_STATUS_LABELS[config.migrationStatus] || config.migrationStatus}
        </Badge>
      </div>

      {activeMigration && (
        <div className="mt-4 rounded-lg border border-border/20 bg-background/70 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <div>
              <span className="font-medium">正在迁移到{BACKEND_LABELS[activeMigration.targetBackendType]}</span>
              <span className="ml-2 text-muted-foreground">迁移期间仍从当前存储读取</span>
            </div>
            <span className="text-muted-foreground">
              {activeMigration.copiedCount}/{activeMigration.totalCount} 个条目
            </span>
          </div>
          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
            <div className="h-full rounded-full bg-primary" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {config.latestMigration?.status === "failed" && (
        <div className="mt-4 flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div className="min-w-0">
            <p className="font-medium">上次迁移未完成</p>
            <p className="mt-1 break-words">{config.latestMigration.errorMessage || "未返回具体错误，请解除失败状态后重新测试。"}</p>
          </div>
        </div>
      )}

      <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <div className="rounded-lg border border-border/20 bg-background/70 p-4">
          <p className="text-xs font-medium tracking-wide text-muted-foreground">当前存储</p>
          <div className="mt-3 flex items-start gap-3">
            <div className="rounded-lg bg-muted p-2 text-foreground">
              {config.backendType === "database" && <Database className="size-5" />}
              {config.backendType === "local" && <HardDrive className="size-5" />}
              {config.backendType === "object_storage" && <Cloud className="size-5" />}
            </div>
            <div className="min-w-0">
              <p className="font-medium">{BACKEND_LABELS[config.backendType]}</p>
              <p className="mt-1 truncate text-sm text-muted-foreground">
                {config.backendType === "database" && "随业务数据库部署，无需额外依赖"}
                {config.backendType === "local" && (config.localPath || "使用服务端默认目录")}
                {config.backendType === "object_storage" && currentStorageName}
              </p>
            </div>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-border/20 bg-card p-3">
              <p className="text-xs text-muted-foreground">工作条目</p>
              <p className="mt-1 text-lg font-semibold">{config.entryCount}</p>
            </div>
            <div className="rounded-lg border border-border/20 bg-card p-3">
              <p className="text-xs text-muted-foreground">内容容量</p>
              <p className="mt-1 text-lg font-semibold">{formatBytes(config.contentBytes)}</p>
            </div>
          </div>
          {canRollback && isAdmin && (
            <Button
              className="mt-4"
              variant="ghost"
              onClick={rollback}
              disabled={rollingBack || config.migrationStatus !== "idle"}
            >
              {rollingBack ? <Loader2 className="animate-spin" /> : <RotateCcw />}
              回滚上次迁移
            </Button>
          )}
        </div>

        <div className="rounded-lg border border-border/20 bg-background/70 p-4">
          <div>
            <h3 className="font-medium">迁移目标</h3>
            <p className="mt-1 text-sm text-muted-foreground">先完成连接与读写测试，再复制、校验并原子切换。</p>
          </div>
          <div className="mt-4 space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="agent-workspace-backend">目标存储</Label>
              <Select
                items={BACKEND_ITEMS}
                value={backendType}
                onValueChange={(value) => value && setBackendType(value as AgentWorkspaceBackend)}
                disabled={controlsDisabled}
              >
                <SelectTrigger id="agent-workspace-backend" className="w-full"><SelectValue placeholder="选择存储类型" /></SelectTrigger>
                <SelectContent><SelectGroup>{BACKEND_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                ))}</SelectGroup></SelectContent>
              </Select>
            </div>

            {backendType === "database" && (
              <div className="flex items-start gap-3 rounded-lg border border-border/20 bg-card p-3 text-sm">
                <Database className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                <p className="text-muted-foreground">数据写入项目数据库，适合默认部署，也便于统一备份和运维。</p>
              </div>
            )}

            {backendType === "local" && (
              <div className="space-y-1.5">
                <Label htmlFor="agent-workspace-local-path">本地目录</Label>
                <div className="relative">
                  <Folder className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    id="agent-workspace-local-path"
                    value={localPath}
                    onChange={(event) => setLocalPath(event.target.value)}
                    placeholder="留空使用服务端默认目录"
                    className="pl-9"
                    disabled={controlsDisabled}
                  />
                </div>
                <p className="text-xs text-muted-foreground">Docker 默认使用 /app/data/agent-workspace，并挂载独立持久卷。</p>
              </div>
            )}

            {backendType === "object_storage" && (
              <div className="space-y-1.5">
                <Label htmlFor="agent-workspace-object-storage">对象存储配置</Label>
                <Select
                  items={objectStorageItems}
                  value={storageConfigId || null}
                  onValueChange={(value) => setStorageConfigId(value || "")}
                  disabled={controlsDisabled}
                >
                  <SelectTrigger id="agent-workspace-object-storage" className="w-full"><SelectValue placeholder="选择已有 S3 兼容配置" /></SelectTrigger>
                  <SelectContent><SelectGroup>{objectStorageItems.map((item) => (
                    <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                  ))}</SelectGroup></SelectContent>
                </Select>
                {objectStorageItems.length === 0 && (
                  <p className="text-xs text-muted-foreground">暂无可用配置，请先在“存储配置”中创建并启用对象存储。</p>
                )}
              </div>
            )}

            <div className="flex flex-wrap justify-end gap-2 border-t border-border/20 pt-4">
              {config.latestMigration?.status === "failed" && isAdmin && (
                <Button variant="outline" onClick={dismissFailure} disabled={dismissing}>
                  {dismissing && <Loader2 className="animate-spin" />}
                  解除失败状态
                </Button>
              )}
              <Button variant="outline" onClick={testTarget} disabled={controlsDisabled || testing || !targetReady}>
                {testing && <Loader2 className="animate-spin" />}
                测试目标
              </Button>
              <Button
                onClick={migrate}
                disabled={controlsDisabled || migrating || !targetReady || targetMatchesCurrent}
              >
                {migrating && <Loader2 className="animate-spin" />}
                迁移并切换
              </Button>
            </div>
            {targetMatchesCurrent && (
              <p className="text-right text-xs text-muted-foreground">当前已经在使用此存储，无需迁移。</p>
            )}
            {!isAdmin && <p className="text-right text-xs text-muted-foreground">只有管理员可以测试或切换工作空间存储。</p>}
          </div>
        </div>
      </div>
    </section>
  );
}
