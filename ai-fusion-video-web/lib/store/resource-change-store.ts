import { create } from "zustand";
import type { ResourceEventPage, ResourceType } from "@/lib/api/resource-events";

export type ResourceArea = "projects" | "scripts" | "assets" | "storyboards";
export type ResourceRevisions = Record<ResourceArea, number>;

const AREAS: Record<ResourceType, readonly ResourceArea[]> = {
  project: ["projects", "scripts", "assets", "storyboards"],
  script: ["scripts"],
  scriptEpisode: ["scripts", "storyboards"],
  scriptScene: ["scripts"],
  asset: ["assets", "storyboards"],
  assetItem: ["assets", "storyboards"],
  storyboard: ["storyboards"],
  storyboardEpisode: ["storyboards", "scripts"],
  storyboardScene: ["storyboards"],
  shot: ["storyboards"],
};

export function changedResourceAreas(page: ResourceEventPage, lastSequence: number): Set<ResourceArea> {
  const areas = new Set<ResourceArea>();
  if (page.reset) return new Set(["projects", "scripts", "assets", "storyboards"]);
  for (const event of page.events) {
    if (event.sequence <= lastSequence) continue;
    const affected = AREAS[event.change.resourceType];
    if (!affected) throw new Error("服务器返回了未登记的资源事件类型");
    for (const area of affected) areas.add(area);
  }
  return areas;
}

interface ResourceChangeState {
  revisions: ResourceRevisions;
  cursor: number;
  scope: string | null;
  apply: (page: ResourceEventPage) => void;
}

export const useResourceChangeStore = create<ResourceChangeState>()((set) => ({
  revisions: { projects: 0, scripts: 0, assets: 0, storyboards: 0 },
  cursor: 0,
  scope: null,
  apply: (page) => set((state) => {
    if (!Number.isSafeInteger(page.cursor) || !Number.isSafeInteger(page.head) || page.cursor < 0 || page.cursor > page.head
      || typeof page.scope !== "string" || !page.scope || (page.reset && page.events.length !== 0)
      || page.events.some((event, index) => !Number.isSafeInteger(event.sequence) || event.sequence <= 0 || event.sequence > page.cursor
        || (index > 0 && event.sequence <= page.events[index - 1].sequence))) {
      throw new Error("资源事件续读协议无效");
    }
    if (!page.reset && page.scope !== state.scope) throw new Error("资源事件作用域已改变，需要重新同步");
    if (!page.reset && page.cursor < state.cursor) return state;
    const revisions = { ...state.revisions };
    for (const area of changedResourceAreas(page, state.cursor)) revisions[area] += 1;
    return { revisions, cursor: page.cursor, scope: page.scope };
  }),
}));
