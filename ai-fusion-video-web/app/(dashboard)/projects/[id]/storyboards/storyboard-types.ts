import type { StoryboardScene, StoryboardItem } from "@/lib/api/storyboard";

export type ViewMode = "table" | "card";

export const SIDEBAR_COLLAPSED_STORAGE_KEY = "fusion-storyboard-sidebar-collapsed";
export const DETAILS_COLLAPSED_STORAGE_KEY = "fusion-storyboard-details-collapsed";

export interface SidebarSelection {
  type: "all" | "episode" | "scene";
  episodeId?: number;
  sceneId?: number;
}

/** 场次及其条目 */
export interface SceneWithItems {
  scene: StoryboardScene;
  items: StoryboardItem[];
}

