import type { CanvasConfig, CanvasNode } from "./types";

export interface CanvasNodeDraft {
  base: CanvasNode;
  title: string;
  config: CanvasConfig;
}

export function isCanvasConfig(value: unknown): value is CanvasConfig {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    && Object.values(value).every((item) => typeof item === "string" || typeof item === "boolean"
      || typeof item === "number" && Number.isFinite(item)
      || Array.isArray(item) && item.every((entry) => typeof entry === "string")
      || isCanvasConfig(item));
}

export function nodeConfigValueEqual(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (Array.isArray(left) || Array.isArray(right)) return Array.isArray(left) && Array.isArray(right)
    && left.length === right.length && left.every((value, index) => nodeConfigValueEqual(value, right[index]));
  if (typeof left !== "object" || left === null || typeof right !== "object" || right === null) return false;
  const a = left as Record<string, unknown>, b = right as Record<string, unknown>;
  return Object.keys(a).length === Object.keys(b).length && Object.keys(a).every((key) => Object.hasOwn(b, key) && nodeConfigValueEqual(a[key], b[key]));
}
export function changedNodeFields(draft: CanvasNodeDraft): string[] {
  return [
    ...(draft.title !== draft.base.title ? ["@title"] : []),
    ...[
      ...new Set([
        ...Object.keys(draft.base.config),
        ...Object.keys(draft.config),
      ]),
    ].filter((key) => !nodeConfigValueEqual(draft.base.config[key], draft.config[key])),
  ];
}
export function nodeDraftValue(
  node: { title: string; config: CanvasConfig },
  key: string,
) {
  return key === "@title" ? node.title : node.config[key];
}
export function nodeDraftConflicts(
  draft: CanvasNodeDraft,
  current: CanvasNode,
) {
  return changedNodeFields(draft).filter(
    (key) =>
      !nodeConfigValueEqual(nodeDraftValue(draft.base, key), nodeDraftValue(current, key)) &&
      !nodeConfigValueEqual(nodeDraftValue(draft, key), nodeDraftValue(current, key)),
  );
}
export function rebaseNodeDraft(
  draft: CanvasNodeDraft,
  current: CanvasNode,
  serverFields: string[],
): CanvasNodeDraft {
  const next = {
    base: current,
    title: current.title,
    config: { ...current.config },
  };
  for (const key of changedNodeFields(draft)) {
    if (serverFields.includes(key)) continue;
    if (key === "@title") next.title = draft.title;
    else if (draft.config[key] === undefined) delete next.config[key];
    else next.config[key] = draft.config[key];
  }
  return next;
}
