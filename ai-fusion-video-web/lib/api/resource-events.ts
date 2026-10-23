import { http } from "./client";

export type ResourceType =
  | "project" | "script" | "scriptEpisode" | "scriptScene"
  | "asset" | "assetItem" | "storyboard" | "storyboardEpisode" | "storyboardScene" | "shot";

export interface ResourceChange {
  eventId: string;
  resourceType: ResourceType;
  resourceId: number;
  projectId: number | null;
  version: number;
  deleted: boolean;
  changedFields: string[];
}

export interface ResourceEventPage {
  cursor: number;
  head: number;
  scope: string;
  reset: boolean;
  hasMore: boolean;
  events: { sequence: number; commandId: string; change: ResourceChange }[];
}

export function readResourceEvents(cursor: number | undefined, scope: string | undefined, signal: AbortSignal) {
  return http.get<never, ResourceEventPage>("/api/resource-events", {
    params: { afterSequence: cursor, scope, limit: 200 },
    signal,
  });
}
