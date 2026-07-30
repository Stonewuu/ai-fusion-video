export type ToolTimelineStatus =
  | "calling"
  | "awaiting_approval"
  | "approved"
  | "rejected"
  | "done"
  | "error"
  | "cancelled";

export type SubTimelineItem =
  | {
      type: "tool";
      id: string;
      name: string;
      arguments: string;
      batchId?: string;
      status: ToolTimelineStatus;
      result?: string | null;
    }
  | { type: "content"; text: string }
  | {
      type: "reasoning";
      text: string;
      startedAtMs?: number;
      durationMs?: number;
    };

export type TimelineItem =
  | {
      type: "tool";
      id: string;
      name: string;
      arguments: string;
      batchId?: string;
      status: ToolTimelineStatus;
      result?: string | null;
      agentName?: string;
      children?: SubTimelineItem[];
    }
  | {
      type: "reasoning";
      text: string;
      startedAtMs?: number;
      durationMs?: number;
    }
  | { type: "content"; text: string };

function isInProgressToolStatus(status: ToolTimelineStatus): boolean {
  return status === "calling"
    || status === "awaiting_approval"
    || status === "approved";
}

function cancelCallingSubTimelineTools(
  children: SubTimelineItem[],
): SubTimelineItem[] {
  let changed = false;
  const next = children.map((child) => {
    if (child.type !== "tool" || !isInProgressToolStatus(child.status)) return child;
    changed = true;
    return { ...child, status: "cancelled" as const };
  });
  return changed ? next : children;
}

export function cancelCallingTimelineTools(
  timeline: TimelineItem[],
): TimelineItem[] {
  let changed = false;
  const next = timeline.map((item) => {
    if (item.type !== "tool") return item;
    const children = item.children
      ? cancelCallingSubTimelineTools(item.children)
      : item.children;
    const status = isInProgressToolStatus(item.status) ? "cancelled" : item.status;
    if (status === item.status && children === item.children) return item;
    changed = true;
    return { ...item, status, children };
  });
  return changed ? next : timeline;
}

function restoreSubTimelineTools(
  current: SubTimelineItem[],
  previous: SubTimelineItem[],
): SubTimelineItem[] {
  const previousTools = new Map(
    previous
      .filter((item): item is Extract<SubTimelineItem, { type: "tool" }> =>
        item.type === "tool")
      .map((item) => [item.id, item]),
  );
  let changed = false;
  const next = current.map((item) => {
    if (item.type !== "tool") return item;
    const previousItem = previousTools.get(item.id);
    if (
      item.status !== "cancelled"
      || !previousItem
      || !isInProgressToolStatus(previousItem.status)
    ) {
      return item;
    }
    changed = true;
    return { ...item, status: previousItem.status };
  });
  return changed ? next : current;
}

export function restoreOptimisticallyCancelledTimelineTools(
  current: TimelineItem[],
  previous: TimelineItem[],
): TimelineItem[] {
  const previousTools = new Map(
    previous
      .filter((item): item is Extract<TimelineItem, { type: "tool" }> =>
        item.type === "tool")
      .map((item) => [item.id, item]),
  );
  let changed = false;
  const next = current.map((item) => {
    if (item.type !== "tool") return item;
    const previousItem = previousTools.get(item.id);
    const children = item.children && previousItem?.children
      ? restoreSubTimelineTools(item.children, previousItem.children)
      : item.children;
    const status = item.status === "cancelled"
      && previousItem
      && isInProgressToolStatus(previousItem.status)
      ? previousItem.status
      : item.status;
    if (status === item.status && children === item.children) return item;
    changed = true;
    return { ...item, status, children };
  });
  return changed ? next : current;
}

export function finishedToolTimelineStatus(
  status: string | undefined,
): Extract<ToolTimelineStatus, "done" | "error" | "cancelled"> {
  if (status === "success" || status === "done") return "done";
  if (status === "error") return "error";
  if (status === "cancelled") return "cancelled";
  throw new Error(`Unsupported finished tool status: ${String(status)}`);
}

export function persistedToolTimelineStatus(status?: string): ToolTimelineStatus {
  if (status === "running") return "calling";
  if (status === "rejected") return "rejected";
  return finishedToolTimelineStatus(status);
}
