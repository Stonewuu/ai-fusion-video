import type { ResourceType } from "@/lib/api/resource-events";
import { RESOURCE_FIELDS } from "./resource-fields";
import type { CanvasResource } from "./types";

export interface CanvasResourceDraft {
  baseline: CanvasResource;
  patch: Record<string, unknown>;
  requestId: string;
}
export function normalizeResourcePatch(
  type: ResourceType,
  patch: Record<string, unknown>,
): Record<string, unknown> {
  const result = { ...patch };
  for (const field of RESOURCE_FIELDS[type]) {
    if (!Object.hasOwn(result, field.key)) continue;
    let value = result[field.key];
    if (field.kind === "json" && typeof value === "string") {
      try {
        value = JSON.parse(value);
      } catch {
        throw new Error(`${field.label}不是有效的 JSON`);
      }
      if (
        value === null ||
        typeof value !== "object" ||
        (field.jsonType === "array"
          ? !Array.isArray(value)
          : Array.isArray(value))
      )
        throw new Error(
          `${field.label}应为${field.jsonType === "array" ? "数组" : "对象"}`,
        );
      result[field.key] = value;
    }
    if (field.kind === "tags" && Array.isArray(value))
      result[field.key] = value
        .map(String)
        .map((item) => item.trim())
        .filter(Boolean);
    if (
      field.required &&
      (value === null || value === undefined || value === "")
    )
      throw new Error(`请填写${field.label}`);
    if (
      field.kind === "number" &&
      value !== null &&
      value !== undefined &&
      (typeof value !== "number" || !Number.isFinite(value))
    )
      throw new Error(`${field.label}必须是有效数字`);
  }
  return result;
}
export function conflictingFields(
  draft: CanvasResourceDraft,
  current: CanvasResource,
): string[] {
  return Object.keys(draft.patch).filter(
    (key) =>
      JSON.stringify(draft.baseline.data[key]) !==
        JSON.stringify(current.data[key]) &&
      JSON.stringify(draft.patch[key]) !== JSON.stringify(current.data[key]),
  );
}

export function acknowledgeResourceDraft(
  current: CanvasResourceDraft,
  submitted: CanvasResourceDraft,
  acknowledged: CanvasResource,
): CanvasResourceDraft {
  const desired = { ...current.baseline.data, ...current.patch };
  const submittedValues = { ...submitted.baseline.data, ...submitted.patch };
  const remaining: Record<string, unknown> = {};
  for (const key of new Set([
    ...Object.keys(current.patch),
    ...Object.keys(submitted.patch),
  ])) {
    if (JSON.stringify(desired[key]) !== JSON.stringify(submittedValues[key]))
      remaining[key] = desired[key];
  }
  return {
    baseline: acknowledged,
    patch: remaining,
    requestId: crypto.randomUUID(),
  };
}
