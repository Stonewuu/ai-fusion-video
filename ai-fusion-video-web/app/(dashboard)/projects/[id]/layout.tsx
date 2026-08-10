"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import { projectApi, type Project } from "@/lib/api/project";
import { toastApiError } from "@/lib/api/toast-api-error";
import { ProjectContext } from "./project-context";

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

  const fetchProject = useCallback(async () => {
    try {
      setLoading(true);
      const proj = await projectApi.get(projectId);
      // 后端 properties 是 JSON 字符串，需要解析为对象
      if (proj && typeof proj.properties === "string") {
        try {
          proj.properties = JSON.parse(proj.properties);
        } catch {
          proj.properties = null;
        }
      }
      setProject(proj);
    } catch (err) {
      console.error("加载项目失败:", err);
      toastApiError(err, "加载项目失败");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchProject();
  }, [fetchProject]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <ProjectContext.Provider value={{ project, loading, refresh: fetchProject }}>
      {children}
    </ProjectContext.Provider>
  );
}
