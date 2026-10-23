"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import { projectApi, type Project } from "@/lib/api/project";
import { getApiErrorStatus } from "@/lib/api/api-error";
import { toastApiError } from "@/lib/api/toast-api-error";
import { ProjectContext } from "./project-context";
import { useResourceInvalidation } from "@/hooks/use-resource-invalidation";

/**
 * 项目详情布局
 * 仅负责加载项目数据并通过 ProjectContext 共享给子页面。
 * 侧边栏导航由全局 SidebarNav 组件处理。
 */
export default function ProjectDetailLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const params = useParams();
  const projectId = Number(params.id);

  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const latestRequest = useRef(0);

  const fetchProject = useCallback(async () => {
    const request = ++latestRequest.current;
    try {
      setLoading(true);
      const proj = await projectApi.get(projectId);
      if (request === latestRequest.current) setProject(proj);
    } catch (err) {
      if (request !== latestRequest.current) return;
      if ([403, 404].includes(getApiErrorStatus(err) ?? 0)) setProject(null);
      console.error("加载项目失败:", err);
      toastApiError(err, "加载项目失败");
    } finally {
      if (request === latestRequest.current) setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void fetchProject();
    return () => { latestRequest.current += 1; };
  }, [fetchProject]);
  useResourceInvalidation("projects", fetchProject);

  if (loading && project?.id !== projectId) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <ProjectContext.Provider value={{ project: project?.id === projectId ? project : null, loading: loading && project?.id !== projectId, refresh: fetchProject }}>
      {children}
    </ProjectContext.Provider>
  );
}
