"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Plus,
  Search,
  FolderKanban,
  MoreHorizontal,
  Clock,
  Film,
  Trash2,
  Loader2,
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { projectApi, type Project } from "@/lib/api/project";
import { CreateProjectDialog } from "@/components/dashboard/create-project-dialog";
import { MainContentFrame } from "@/components/dashboard/main-content-frame";

// 动画
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number],
    },
  },
};

// 渐变色循环
const gradients = [
  "from-blue-500/10 via-transparent to-transparent",
  "from-purple-500/10 via-transparent to-transparent",
  "from-green-500/10 via-transparent to-transparent",
  "from-orange-500/10 via-transparent to-transparent",
  "from-pink-500/10 via-transparent to-transparent",
  "from-cyan-500/10 via-transparent to-transparent",
];

const iconColors = [
  "text-blue-400",
  "text-purple-400",
  "text-green-400",
  "text-orange-400",
  "text-pink-400",
  "text-cyan-400",
];

function formatTime(iso: string) {
  const d = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "刚刚";
  if (mins < 60) return `${mins} 分钟前`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} 天前`;
  return d.toLocaleDateString("zh-CN");
}

export default function ProjectsPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const fetchProjects = useCallback(async () => {
    try {
      setLoading(true);
      const data = await projectApi.list();
      setProjects(data);
    } catch (err) {
      console.error("获取项目列表失败:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  const handleDelete = async (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    if (!confirm("确定要删除该项目吗？此操作不可撤销。")) return;
    setDeletingId(id);
    try {
      await projectApi.delete(id);
      await fetchProjects();
    } catch (err) {
      console.error("删除项目失败:", err);
    } finally {
      setDeletingId(null);
    }
  };

  const filtered = projects.filter(
    (p) =>
      p.name.toLowerCase().includes(search.toLowerCase()) ||
      p.description?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <MainContentFrame>
      <motion.div
        className="max-w-[1200px]"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* 页面标题区域 */}
        <motion.div
          variants={itemVariants}
          className="flex items-center justify-between mb-8"
        >
          <div>
            <h1 className="text-3xl font-bold tracking-tight">项目管理</h1>
            <p className="text-muted-foreground mt-1">
              管理你的所有视频创作项目
            </p>
          </div>
          <button
            onClick={() => setShowCreate(true)}
            className={cn(
              "flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium",
              "bg-linear-to-r from-blue-600 to-purple-600",
              "text-white shadow-lg shadow-blue-500/20",
              "hover:shadow-blue-500/30 hover:scale-[1.02]",
              "active:scale-[0.98] transition-all duration-200"
            )}
          >
            <Plus className="h-4 w-4" />
            新建项目
          </button>
        </motion.div>

        {/* 搜索栏 */}
        <motion.div variants={itemVariants} className="mb-6">
          <div
            className={cn(
              "flex items-center gap-3 px-4 py-3 rounded-xl",
              "border border-border/30 bg-card/50 backdrop-blur-sm"
            )}
          >
            <Search className="h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="搜索项目名称或描述..."
              className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground/60"
            />
          </div>
        </motion.div>

        {/* 加载状态 */}
        {loading ? (
          <motion.div
            variants={itemVariants}
            className="flex items-center justify-center py-20"
          >
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            <span className="ml-2 text-muted-foreground text-sm">
              加载中...
            </span>
          </motion.div>
        ) : filtered.length === 0 ? (
          /* 空状态 */
          <motion.div variants={itemVariants} className="mt-8">
            <div
              onClick={() => setShowCreate(true)}
              className={cn(
                "rounded-xl border border-dashed border-border/40 p-16",
                "flex flex-col items-center justify-center text-center",
                "bg-card/20 hover:border-border/60 transition-colors cursor-pointer"
              )}
            >
              <div className="h-14 w-14 rounded-xl bg-blue-500/10 flex items-center justify-center mb-4">
                <Plus className="h-7 w-7 text-blue-400" />
              </div>
              <p className="text-muted-foreground text-sm">
                {projects.length === 0
                  ? "还没有项目，点击创建第一个视频项目"
                  : "没有找到匹配的项目"}
              </p>
            </div>
          </motion.div>
        ) : (
          /* 项目卡片列表 */
          <motion.div
            variants={itemVariants}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5"
          >
            {filtered.map((project, idx) => (
              <motion.div
                key={project.id}
                whileHover={{ y: -4, transition: { duration: 0.2 } }}
                onClick={() => router.push(`/projects/${project.id}`)}
                className={cn(
                  "group relative rounded-xl border border-border/30 p-5 cursor-pointer",
                  "bg-card/50 backdrop-blur-sm",
                  "hover:border-border/50 transition-all duration-300"
                )}
              >
                {/* 渐变背景 */}
                <div
                  className={cn(
                    "absolute inset-0 rounded-xl bg-linear-to-br opacity-0 group-hover:opacity-100 transition-opacity duration-500",
                    gradients[idx % gradients.length]
                  )}
                />

                <div className="relative z-10">
                  {/* 标题行 */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <FolderKanban
                        className={cn(
                          "h-4 w-4",
                          iconColors[idx % iconColors.length]
                        )}
                      />
                      <h3 className="font-semibold text-base line-clamp-1">
                        {project.name}
                      </h3>
                    </div>
                    <button
                      onClick={(e) => handleDelete(e, project.id)}
                      disabled={deletingId === project.id}
                      className="p-1.5 rounded-lg hover:bg-destructive/10 opacity-0 group-hover:opacity-100 transition-all"
                    >
                      {deletingId === project.id ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                      ) : (
                        <Trash2 className="h-3.5 w-3.5 text-muted-foreground hover:text-destructive" />
                      )}
                    </button>
                  </div>

                  {/* 描述 */}
                  <p className="text-sm text-muted-foreground mb-4 line-clamp-2 min-h-[2.5rem]">
                    {project.description || "暂无描述"}
                  </p>

                  {/* 元信息 */}
                  <div className="flex items-center justify-between">
                    <span
                      className={cn(
                        "text-xs px-2.5 py-1 rounded-lg border",
                        "text-blue-400 bg-blue-500/10 border-blue-500/20"
                      )}
                    >
                      进行中
                    </span>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {formatTime(project.updateTime)}
                      </span>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </motion.div>

      <CreateProjectDialog
        open={showCreate}
        onClose={() => setShowCreate(false)}
        onCreated={fetchProjects}
      />
    </MainContentFrame>
  );
}
