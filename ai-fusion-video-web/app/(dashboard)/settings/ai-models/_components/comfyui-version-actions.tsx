"use client";

import { useState } from "react";
import { Loader2, Rocket, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { toastApiError } from "@/lib/api/toast-api-error";
import {
  comfyUiWorkflowApi,
  type ComfyUiWorkflow,
  type ComfyUiWorkflowVersion,
} from "@/lib/api/comfyui-workflow";
import { useAuthStore } from "@/lib/store/auth-store";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import { ComfyUiTestPanel } from "./comfyui-test-panel";

const testStatus: Record<number, string> = {
  0: "未运行",
  1: "已成功",
  2: "失败",
  3: "进行中",
  4: "已取消",
  5: "需要核实",
};

export function ComfyUiVersionActions({
  workflow,
  version,
  onUpdated,
}: {
  workflow: ComfyUiWorkflow;
  version: ComfyUiWorkflowVersion | null;
  onUpdated: () => Promise<void> | void;
}) {
  const [running, setRunning] = useState<"validate" | "publish" | null>(null);
  const userId = useAuthStore((state) => state.user?.id);
  const scope = useResourceChangeStore((state) => state.scope);
  const validate = async () => {
    if (!version) return;
    setRunning("validate");
    try {
      const result = await comfyUiWorkflowApi.validateVersion(version.id);
      if (result.valid)
        toast.success("在线校验通过", {
          description: `已检查 ${result.checkedNodeCount} 个节点`,
        });
      else toast.error("在线校验未通过", { description: result.message });
    } catch (error) {
      toastApiError(error, "在线校验失败");
    } finally {
      setRunning(null);
      try {
        await onUpdated();
      } catch (error) {
        toastApiError(error, "刷新校验结果失败");
      }
    }
  };
  const publish = async () => {
    if (!version) return;
    setRunning("publish");
    try {
      await comfyUiWorkflowApi.publish(workflow.id, version.id);
      toast.success(
        workflow.activeVersionId ? "工作流版本已切换" : "工作流已发布",
      );
      await onUpdated();
    } catch (error) {
      toastApiError(error, "发布工作流失败");
    } finally {
      setRunning(null);
    }
  };
  if (!version)
    return (
      <p className="text-sm text-muted-foreground">
        先保存工作流版本，再在线校验、试运行和发布。
      </p>
    );
  const isActive = workflow.activeVersionId === version.id;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-2">
          <p className="text-sm font-medium">
            版本 {version.versionNo} ·{" "}
            {version.published ? "已发布" : "已保存的草稿"}
          </p>
          <p className="text-xs text-muted-foreground">
            在线校验：
            {version.validationStatus === 1
              ? "已通过"
              : version.validationStatus === 2
                ? "未通过"
                : "未校验"}{" "}
            · 试运行：{testStatus[version.testStatus] ?? "未知状态"}
          </p>
          {version.validationMessage && (
            <p className="text-xs text-muted-foreground">
              {version.validationMessage}
            </p>
          )}
          {version.testMessage && (
            <p className="text-xs text-muted-foreground">
              {version.testMessage}
            </p>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => void validate()}
            disabled={running !== null}
          >
            {running === "validate" ? (
              <Loader2 className="animate-spin motion-reduce:animate-none" />
            ) : (
              <ShieldCheck />
            )}
            在线校验
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void publish()}
            disabled={
              running !== null ||
              version.validationStatus !== 1 ||
              version.testStatus !== 1 ||
              isActive
            }
          >
            {running === "publish" ? (
              <Loader2 className="animate-spin motion-reduce:animate-none" />
            ) : (
              <Rocket />
            )}
            {isActive
              ? "当前已发布"
              : version.published
                ? "回滚到此版本"
                : "发布版本"}
          </Button>
        </div>
      </div>
      <ComfyUiTestPanel
        key={`${userId}:${scope}:${version.id}`}
        version={version}
        disabled={running !== null}
        onUpdated={onUpdated}
      />
    </div>
  );
}
