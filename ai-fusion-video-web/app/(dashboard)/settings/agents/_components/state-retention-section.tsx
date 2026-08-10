"use client";

import { useEffect, useState } from "react";
import { Database, Save } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  agentConfigApi,
  type AgentStateCleanupPolicy,
} from "@/lib/api/agent-config";
import { toastApiError } from "@/lib/api/toast-api-error";
import { settingsTypography } from "../../_shared";

interface StateRetentionSectionProps {
  policy: AgentStateCleanupPolicy;
  isAdmin: boolean;
  onSaved: (policy: AgentStateCleanupPolicy) => void;
}

function formatSchedule(value?: string | null): string {
  if (!value) return "尚未执行";
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

export function StateRetentionSection({
  policy,
  isAdmin,
  onSaved,
}: StateRetentionSectionProps) {
  const [cleanupIntervalDays, setCleanupIntervalDays] = useState(
    policy.cleanupIntervalDays,
  );
  const [retentionDays, setRetentionDays] = useState(policy.retentionDays);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setCleanupIntervalDays(policy.cleanupIntervalDays);
    setRetentionDays(policy.retentionDays);
  }, [policy]);

  const save = async () => {
    if (
      !Number.isInteger(cleanupIntervalDays)
      || cleanupIntervalDays < 1
      || cleanupIntervalDays > 365
    ) {
      toast.error("清理周期必须是 1–365 天的整数");
      return;
    }
    if (
      !Number.isInteger(retentionDays)
      || retentionDays < 1
      || retentionDays > 3650
    ) {
      toast.error("保留期限必须是 1–3650 天的整数");
      return;
    }
    setSaving(true);
    try {
      const saved = await agentConfigApi.saveStateCleanupPolicy({
        cleanupIntervalDays,
        retentionDays,
      });
      onSaved(saved);
      toast.success("AgentState 清理配置已保存");
    } catch (error) {
      toastApiError(error, "保存 AgentState 清理配置失败");
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="rounded-xl border border-border/30 bg-card/50 p-4 backdrop-blur-sm">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="rounded-xl bg-primary/10 p-2 text-primary">
            <Database className="size-5" />
          </div>
          <div>
            <h2 className={settingsTypography.sectionTitle}>AgentState 生命周期</h2>
            <p className={settingsTypography.sectionDescription}>
              定期删除长期未活动的模型会话状态；消息记录仍会保留，但过期会话不能继续执行。
            </p>
          </div>
        </div>
        {isAdmin && (
          <Button type="button" onClick={save} disabled={saving}>
            <Save />
            {saving ? "保存中…" : "保存"}
          </Button>
        )}
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="agent-state-cleanup-interval" className={settingsTypography.fieldLabel}>清理周期（天）</Label>
          <Input
            id="agent-state-cleanup-interval"
            type="number"
            min={1}
            max={365}
            step={1}
            value={cleanupIntervalDays}
            disabled={!isAdmin || saving}
            onChange={(event) => setCleanupIntervalDays(Number(event.target.value))}
          />
          <p className="text-xs text-muted-foreground">
            每隔 {cleanupIntervalDays || "—"} 天检查一次到期状态。
          </p>
        </div>
        <div className="space-y-2">
          <Label htmlFor="agent-state-retention-days" className={settingsTypography.fieldLabel}>保留期限（天）</Label>
          <Input
            id="agent-state-retention-days"
            type="number"
            min={1}
            max={3650}
            step={1}
            value={retentionDays}
            disabled={!isAdmin || saving}
            onChange={(event) => setRetentionDays(Number(event.target.value))}
          />
          <p className="text-xs text-muted-foreground">
            清理超过 {retentionDays || "—"} 天未活动的 AgentState。
          </p>
        </div>
      </div>

      <div className="mt-4 grid gap-3 rounded-lg border border-border/20 bg-background/70 p-3 text-xs text-muted-foreground sm:grid-cols-2">
        <p>下次计划：{formatSchedule(policy.nextCleanupAt)}</p>
        <p>上次完成：{formatSchedule(policy.lastCleanupAt)}</p>
      </div>
      {!isAdmin && (
        <p className="mt-3 text-xs text-muted-foreground">仅管理员可以修改全局清理策略。</p>
      )}
    </section>
  );
}
