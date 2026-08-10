<<<<<<< Updated upstream
"use client";

import { useState } from "react";
import {
  Ban,
  Braces,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock3,
  ListTree,
  Loader2,
  ShieldAlert,
  Wrench,
  XCircle,
} from "lucide-react";
import { ToolResultDisplay } from "@/components/dashboard/notification-panel/results";
import { Button } from "@/components/ui/button";
import type { ToolTimelineStatus } from "@/lib/store/pipeline-store";
import { cn } from "@/lib/utils";
import { getToolDisplayName } from "./ai-task-display";
import { formatToolResultJson } from "./tool-call-argument-model";
import { ToolCallArguments } from "./tool-call-arguments";
import { ToolCallJson } from "./tool-call-json";
import { ToolResultViewport } from "./tool-result-viewport";
import { useToolConfirmationCountdown } from "./tool-confirmation-countdown";

export interface ToolCallApproval {
  decision?: "approve" | "deny";
  submitting: boolean;
  showActions: boolean;
  showCountdown?: boolean;
  expiresAt: string;
  onApprove: () => void;
  onReject: () => void;
}

export interface ToolCallCardItem {
  toolCallId: string;
  toolName: string;
  argumentsText: string;
  status: ToolTimelineStatus;
  result?: string | null;
}

interface ToolCallCardProps {
  items: ToolCallCardItem[];
  approval?: ToolCallApproval;
}

function statusContent(
  status: ToolTimelineStatus,
  approval: ToolCallApproval | undefined,
) {
  if (status === "awaiting_approval" && approval?.decision === "approve") {
    return approval.submitting
      ? { label: "正在允许", icon: Loader2, className: "animate-spin text-primary motion-reduce:animate-none" }
      : { label: "已选允许", icon: CheckCircle2, className: "text-primary" };
  }
  if (status === "awaiting_approval" && approval?.decision === "deny") {
    return approval.submitting
      ? { label: "正在拒绝", icon: Loader2, className: "animate-spin text-muted-foreground motion-reduce:animate-none" }
      : { label: "已选拒绝", icon: Ban, className: "text-muted-foreground" };
  }
  switch (status) {
    case "preparing":
      return { label: "正在发起工具调用", icon: Loader2, className: "animate-spin text-primary motion-reduce:animate-none" };
    case "calling":
      return { label: "执行中", icon: Loader2, className: "animate-spin text-primary motion-reduce:animate-none" };
    case "awaiting_approval":
      return { label: "待确认", icon: ShieldAlert, className: "text-foreground" };
    case "approved":
      return { label: "已允许，执行中", icon: Loader2, className: "animate-spin text-primary motion-reduce:animate-none" };
    case "rejected":
      return { label: "已拒绝", icon: Ban, className: "text-muted-foreground" };
    case "expired":
      return { label: "审批超时，未执行", icon: Clock3, className: "text-muted-foreground" };
    case "done":
      return { label: "已完成", icon: CheckCircle2, className: "text-primary" };
    case "error":
      return { label: "执行失败", icon: XCircle, className: "text-destructive" };
    case "cancelled":
      return { label: "已取消", icon: Ban, className: "text-muted-foreground" };
  }
}

function ToolCallItem({
  item,
  approval,
}: {
  item: ToolCallCardItem;
  approval?: ToolCallApproval;
}) {
  const result = item.result;
  const hasResult = result !== undefined
    && item.status !== "rejected"
    && item.status !== "expired"
    && item.status !== "cancelled";
  const argumentsReady = item.status !== "preparing"
    && item.argumentsText.trim().length > 0;
  const jsonViewReady = hasResult || argumentsReady;
  const [argumentsExpanded, setArgumentsExpanded] = useState(
    item.status === "awaiting_approval" && !hasResult,
  );
  const [jsonView, setJsonView] = useState(false);
  const showCountdown = approval?.showCountdown !== false;
  const approvalCountdown = useToolConfirmationCountdown(
    approval?.expiresAt,
    showCountdown,
  );
  const approvalExpired = item.status === "awaiting_approval"
    && approvalCountdown?.expired === true;

  const status = approvalExpired
    ? { label: "审批已超时", icon: Clock3, className: "text-muted-foreground" }
    : statusContent(item.status, approval);
  const StatusIcon = status.icon;
  const submissionPending = approval?.submitting === true;

  return (
    <section
      className={cn(
        "group/tool-call overflow-hidden rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm",
        item.status === "error" && "border-destructive/30",
      )}
      aria-live={item.status === "preparing"
        || item.status === "calling"
        || item.status === "approved"
        ? "polite"
        : undefined}
    >
      <div className="flex min-h-11 items-center gap-2 px-3 py-2.5">
        <Wrench className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
        <h3 className="min-w-0 truncate text-sm font-medium">
          {getToolDisplayName(item.toolName)}
        </h3>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          className={cn(
            "pointer-events-none ml-auto opacity-0 transition-opacity duration-150 group-focus-within/tool-call:pointer-events-auto group-focus-within/tool-call:opacity-100 group-hover/tool-call:pointer-events-auto group-hover/tool-call:opacity-100 motion-reduce:transition-none",
            !jsonViewReady && "hidden",
          )}
          onClick={() => setJsonView((current) => !current)}
        >
          {jsonView ? <ListTree /> : <Braces />}
          {jsonView ? "返回友好视图" : "查看 JSON"}
        </Button>
        <span className="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
          <StatusIcon className={cn("size-3.5", status.className)} aria-hidden="true" />
          {status.label}
          {item.status === "awaiting_approval"
            && approvalCountdown
            && showCountdown
            && !approvalCountdown.expired
            ? ` · 剩余 ${approvalCountdown.label}`
            : null}
        </span>
      </div>

      {hasResult ? (
        <div className="border-t border-border/20 px-3 py-3">
          {jsonView ? (
            <ToolCallJson
              code={formatToolResultJson(result)}
              label="执行结果 JSON"
            />
          ) : (
            <ToolResultViewport>
              {(expanded) => result === null ? (
                <p className="text-xs text-muted-foreground">
                  工具未返回结果
                </p>
              ) : (
                <ToolResultDisplay
                  toolName={item.toolName}
                  content={result}
                  expanded={expanded}
                />
              )}
            </ToolResultViewport>
          )}
        </div>
      ) : null}

      {argumentsReady ? (
        <div className="border-t border-border/20">
          <Button
            type="button"
            variant="ghost"
            size="xs"
            className="m-2"
            onClick={() => setArgumentsExpanded((current) => !current)}
            aria-expanded={argumentsExpanded}
          >
            {argumentsExpanded ? <ChevronDown /> : <ChevronRight />}
            <ListTree />
            调用参数
          </Button>
          {argumentsExpanded ? (
            <div className="px-3 pb-3">
              <ToolCallArguments
                argumentsText={item.argumentsText}
                toolName={item.toolName}
                view={jsonView ? "json" : "friendly"}
              />
            </div>
          ) : null}
        </div>
      ) : null}

      {approval?.showActions && approvalExpired ? (
        <div className="flex items-center gap-2 border-t border-border/20 px-3 py-2.5 text-xs text-muted-foreground">
          <Clock3 className="size-3.5 shrink-0" aria-hidden="true" />
          审批时间已结束，系统将按未同意处理
        </div>
      ) : null}

      {approval?.showActions && !approvalExpired ? (
        <div className="flex items-center justify-end gap-2 border-t border-border/20 px-3 py-2.5">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={submissionPending}
            onClick={approval.onReject}
          >
            {approval.submitting && approval.decision === "deny" ? (
              <Loader2 className="animate-spin motion-reduce:animate-none" />
            ) : null}
            {approval.decision === "deny" && !submissionPending ? "已选拒绝" : "拒绝"}
          </Button>
          <Button
            type="button"
            variant="default"
            size="sm"
            disabled={submissionPending}
            onClick={approval.onApprove}
          >
            {approval.submitting && approval.decision === "approve" ? (
              <Loader2 className="animate-spin motion-reduce:animate-none" />
            ) : null}
            {approval.decision === "approve" && !submissionPending
              ? "已选允许"
              : "允许执行"}
          </Button>
        </div>
      ) : null}
    </section>
  );
}

export function ToolCallCard({ items, approval }: ToolCallCardProps) {
  if (items.length === 0) {
    throw new Error("Tool call card requires at least one item");
  }
  if (new Set(items.map((item) => item.toolCallId)).size !== items.length) {
    throw new Error("Tool call card contains duplicate toolCallIds");
  }
  const awaitingItems = items.filter((item) => item.status === "awaiting_approval");
  if (approval && (items.length !== 1 || awaitingItems.length !== 1)) {
    throw new Error("Approval controls must target exactly one awaiting tool call");
  }
  if (!approval && awaitingItems.length > 0) {
    throw new Error("Awaiting tool call card has no approval controls");
  }
  if (approval?.submitting && !approval.decision) {
    throw new Error("Submitting tool approval has no decision");
  }

  return (
    <div className="flex flex-col gap-2">
      {items.map((item) => (
        <ToolCallItem
          key={`${item.toolCallId}-${item.result === undefined ? "pending" : "settled"}`}
          item={item}
          approval={approval}
        />
      ))}
    </div>
  );
}
=======
"use client";

import { useState } from "react";
import {
  Ban,
  Braces,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock3,
  ListTree,
  Loader2,
  ShieldAlert,
  Wrench,
  XCircle,
} from "lucide-react";
import { ToolResultDisplay } from "@/components/dashboard/notification-panel/results";
import { Button } from "@/components/ui/button";
import type { ToolTimelineStatus } from "@/lib/store/pipeline-store";
import { cn } from "@/lib/utils";
import { getToolDisplayName } from "./ai-task-display";
import { formatToolResultJson } from "./tool-call-argument-model";
import { ToolCallArguments } from "./tool-call-arguments";
import { ToolCallJson } from "./tool-call-json";
import { ToolResultViewport } from "./tool-result-viewport";
import { useToolConfirmationCountdown } from "./tool-confirmation-countdown";

export interface ToolCallApproval {
  decision?: "approve" | "deny";
  submitting: boolean;
  showActions: boolean;
  showCountdown?: boolean;
  expiresAt: string;
  onApprove: () => void;
  onReject: () => void;
}

export interface ToolCallCardItem {
  toolCallId: string;
  toolName: string;
  argumentsText: string;
  status: ToolTimelineStatus;
  result?: string | null;
}

interface ToolCallCardProps {
  items: ToolCallCardItem[];
  approval?: ToolCallApproval;
}

function statusContent(
  status: ToolTimelineStatus,
  approval: ToolCallApproval | undefined,
) {
  if (status === "awaiting_approval" && approval?.decision === "approve") {
    return approval.submitting
      ? { label: "正在允许", icon: Loader2, className: "animate-spin text-primary motion-reduce:animate-none" }
      : { label: "已选允许", icon: CheckCircle2, className: "text-primary" };
  }
  if (status === "awaiting_approval" && approval?.decision === "deny") {
    return approval.submitting
      ? { label: "正在拒绝", icon: Loader2, className: "animate-spin text-muted-foreground motion-reduce:animate-none" }
      : { label: "已选拒绝", icon: Ban, className: "text-muted-foreground" };
  }
  switch (status) {
    case "preparing":
      return { label: "正在发起工具调用", icon: Loader2, className: "animate-spin text-primary motion-reduce:animate-none" };
    case "calling":
      return { label: "执行中", icon: Loader2, className: "animate-spin text-primary motion-reduce:animate-none" };
    case "awaiting_approval":
      return { label: "待确认", icon: ShieldAlert, className: "text-foreground" };
    case "approved":
      return { label: "已允许，执行中", icon: Loader2, className: "animate-spin text-primary motion-reduce:animate-none" };
    case "rejected":
      return { label: "已拒绝", icon: Ban, className: "text-muted-foreground" };
    case "expired":
      return { label: "审批超时，未执行", icon: Clock3, className: "text-muted-foreground" };
    case "done":
      return { label: "已完成", icon: CheckCircle2, className: "text-primary" };
    case "error":
      return { label: "执行失败", icon: XCircle, className: "text-destructive" };
    case "cancelled":
      return { label: "已取消", icon: Ban, className: "text-muted-foreground" };
  }
}

function ToolCallItem({
  item,
  approval,
}: {
  item: ToolCallCardItem;
  approval?: ToolCallApproval;
}) {
  const result = item.result;
  const hasResult = result !== undefined
    && item.status !== "rejected"
    && item.status !== "expired"
    && item.status !== "cancelled";
  const argumentsReady = item.status !== "preparing";
  const [argumentsExpanded, setArgumentsExpanded] = useState(
    item.status === "awaiting_approval" && !hasResult,
  );
  const [jsonView, setJsonView] = useState(false);
  const showCountdown = approval?.showCountdown !== false;
  const approvalCountdown = useToolConfirmationCountdown(
    approval?.expiresAt,
    showCountdown,
  );
  const approvalExpired = item.status === "awaiting_approval"
    && approvalCountdown?.expired === true;

  const status = approvalExpired
    ? { label: "审批已超时", icon: Clock3, className: "text-muted-foreground" }
    : statusContent(item.status, approval);
  const StatusIcon = status.icon;
  const submissionPending = approval?.submitting === true;

  return (
    <section
      className={cn(
        "group/tool-call overflow-hidden rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm",
        item.status === "error" && "border-destructive/30",
      )}
      aria-live={item.status === "preparing"
        || item.status === "calling"
        || item.status === "approved"
        ? "polite"
        : undefined}
    >
      <div className="flex min-h-11 items-center gap-2 px-3 py-2.5">
        <Wrench className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
        <h3 className="min-w-0 truncate text-sm font-medium">
          {getToolDisplayName(item.toolName)}
        </h3>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          className={cn(
            "pointer-events-none ml-auto opacity-0 transition-opacity duration-150 group-focus-within/tool-call:pointer-events-auto group-focus-within/tool-call:opacity-100 group-hover/tool-call:pointer-events-auto group-hover/tool-call:opacity-100 motion-reduce:transition-none",
            !argumentsReady && "hidden",
          )}
          onClick={() => setJsonView((current) => !current)}
        >
          {jsonView ? <ListTree /> : <Braces />}
          {jsonView ? "返回友好视图" : "查看 JSON"}
        </Button>
        <span className="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
          <StatusIcon className={cn("size-3.5", status.className)} aria-hidden="true" />
          {status.label}
          {item.status === "awaiting_approval"
            && approvalCountdown
            && showCountdown
            && !approvalCountdown.expired
            ? ` · 剩余 ${approvalCountdown.label}`
            : null}
        </span>
      </div>

      {hasResult ? (
        <div className="border-t border-border/20 px-3 py-3">
          {jsonView ? (
            <ToolCallJson
              code={formatToolResultJson(result)}
              label="执行结果 JSON"
            />
          ) : (
            <ToolResultViewport>
              {(expanded) => result === null ? (
                <p className="text-xs text-muted-foreground">
                  工具未返回结果
                </p>
              ) : (
                <ToolResultDisplay
                  toolName={item.toolName}
                  content={result}
                  expanded={expanded}
                />
              )}
            </ToolResultViewport>
          )}
        </div>
      ) : null}

      {argumentsReady ? (
        <div className="border-t border-border/20">
          <Button
            type="button"
            variant="ghost"
            size="xs"
            className="m-2"
            onClick={() => setArgumentsExpanded((current) => !current)}
            aria-expanded={argumentsExpanded}
          >
            {argumentsExpanded ? <ChevronDown /> : <ChevronRight />}
            <ListTree />
            调用参数
          </Button>
          {argumentsExpanded ? (
            <div className="px-3 pb-3">
              <ToolCallArguments
                argumentsText={item.argumentsText}
                toolName={item.toolName}
                view={jsonView ? "json" : "friendly"}
              />
            </div>
          ) : null}
        </div>
      ) : null}

      {approval?.showActions && approvalExpired ? (
        <div className="flex items-center gap-2 border-t border-border/20 px-3 py-2.5 text-xs text-muted-foreground">
          <Clock3 className="size-3.5 shrink-0" aria-hidden="true" />
          审批时间已结束，系统将按未同意处理
        </div>
      ) : null}

      {approval?.showActions && !approvalExpired ? (
        <div className="flex items-center justify-end gap-2 border-t border-border/20 px-3 py-2.5">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={submissionPending}
            onClick={approval.onReject}
          >
            {approval.submitting && approval.decision === "deny" ? (
              <Loader2 className="animate-spin motion-reduce:animate-none" />
            ) : null}
            {approval.decision === "deny" && !submissionPending ? "已选拒绝" : "拒绝"}
          </Button>
          <Button
            type="button"
            variant="default"
            size="sm"
            disabled={submissionPending}
            onClick={approval.onApprove}
          >
            {approval.submitting && approval.decision === "approve" ? (
              <Loader2 className="animate-spin motion-reduce:animate-none" />
            ) : null}
            {approval.decision === "approve" && !submissionPending
              ? "已选允许"
              : "允许执行"}
          </Button>
        </div>
      ) : null}
    </section>
  );
}

export function ToolCallCard({ items, approval }: ToolCallCardProps) {
  if (items.length === 0) {
    throw new Error("Tool call card requires at least one item");
  }
  if (new Set(items.map((item) => item.toolCallId)).size !== items.length) {
    throw new Error("Tool call card contains duplicate toolCallIds");
  }
  const awaitingItems = items.filter((item) => item.status === "awaiting_approval");
  if (approval && (items.length !== 1 || awaitingItems.length !== 1)) {
    throw new Error("Approval controls must target exactly one awaiting tool call");
  }
  if (!approval && awaitingItems.length > 0) {
    throw new Error("Awaiting tool call card has no approval controls");
  }
  if (approval?.submitting && !approval.decision) {
    throw new Error("Submitting tool approval has no decision");
  }

  return (
    <div className="flex flex-col gap-2">
      {items.map((item) => (
        <ToolCallItem
          key={`${item.toolCallId}-${item.result === undefined ? "pending" : "settled"}`}
          item={item}
          approval={approval}
        />
      ))}
    </div>
  );
}
>>>>>>> Stashed changes
