"use client";

import { CheckCheck, Clock3, Loader2, ShieldCheck } from "lucide-react";
import { useToolConfirmationCountdown } from "@/components/dashboard/shared/tool-confirmation-countdown";
import { Button } from "@/components/ui/button";

interface ToolConfirmationBatchBarProps {
  toolCallIds: string[];
  decisions: Readonly<Record<string, boolean>>;
  submitting: boolean;
  showActions: boolean;
  expiresAt: string;
  onDecision: (approved: boolean) => void;
}

function selectedBatchDecision(
  toolCallIds: string[],
  decisions: Readonly<Record<string, boolean>>,
): boolean | undefined {
  if (toolCallIds.some((toolCallId) =>
    !Object.prototype.hasOwnProperty.call(decisions, toolCallId))) {
    return undefined;
  }
  const firstDecision = decisions[toolCallIds[0]];
  return toolCallIds.every((toolCallId) => decisions[toolCallId] === firstDecision)
    ? firstDecision
    : undefined;
}

export function ToolConfirmationBatchBar({
  toolCallIds,
  decisions,
  submitting,
  showActions,
  expiresAt,
  onDecision,
}: ToolConfirmationBatchBarProps) {
  const countdown = useToolConfirmationCountdown(expiresAt);
  if (toolCallIds.length < 2) {
    throw new Error("Batch approval requires at least two tool calls");
  }
  if (!countdown) {
    throw new Error("Batch approval requires an expiry");
  }
  const expired = countdown.expired;
  const selectedCount = toolCallIds.filter((toolCallId) =>
    Object.prototype.hasOwnProperty.call(decisions, toolCallId)).length;
  const batchDecision = selectedBatchDecision(toolCallIds, decisions);

  if (!showActions && !expired) return null;

  return (
    <section
      className="rounded-2xl border border-border/40 bg-popover/80 p-3 shadow-xl backdrop-blur-xl"
      aria-live="polite"
      data-assistant-batch-approval
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="flex min-w-0 flex-1 items-center gap-2.5">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
            {expired ? (
              <Clock3 className="size-4" />
            ) : submitting ? (
              <Loader2 className="size-4 animate-spin motion-reduce:animate-none" />
            ) : (
              <ShieldCheck className="size-4" />
            )}
          </span>
          <div className="min-w-0">
            <h3 className="text-sm font-medium">批量审批</h3>
            <p className="truncate text-xs text-muted-foreground">
              {expired
                ? "审批时间已结束，系统将按未同意处理"
                : submitting
                  ? `正在提交 ${toolCallIds.length} 个工具的审批决定`
                : `${toolCallIds.length} 个工具等待确认${selectedCount > 0
                  ? ` · 已选择 ${selectedCount}/${toolCallIds.length}`
                  : ""} · 剩余 ${countdown.label}`}
            </p>
          </div>
        </div>

        {!expired ? (
          <div className="flex shrink-0 items-center justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={submitting}
              data-assistant-interactive="true"
              onClick={() => onDecision(false)}
            >
              {submitting && batchDecision === false ? (
                <Loader2 className="animate-spin motion-reduce:animate-none" />
              ) : null}
              全部拒绝
            </Button>
            <Button
              type="button"
              variant="default"
              size="sm"
              disabled={submitting}
              data-assistant-interactive="true"
              onClick={() => onDecision(true)}
            >
              {submitting && batchDecision === true ? (
                <Loader2 className="animate-spin motion-reduce:animate-none" />
              ) : (
                <CheckCheck />
              )}
              全部允许
            </Button>
          </div>
        ) : null}
      </div>
    </section>
  );
}
