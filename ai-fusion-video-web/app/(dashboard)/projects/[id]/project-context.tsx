"use client";

import { createContext, useContext } from "react";
import type { Project } from "@/lib/api/project";

// ========== 项目上下文 ==========

interface ProjectContextValue {
  project: Project | null;
  loading: boolean;
  refresh: () => Promise<void>;
}

export const ProjectContext = createContext<ProjectContextValue>({
  project: null,
  loading: true,
  refresh: async () => {},
});

/** 在项目详情子页面中获取当前项目数据 */
export function useProject() {
  return useContext(ProjectContext);
}
