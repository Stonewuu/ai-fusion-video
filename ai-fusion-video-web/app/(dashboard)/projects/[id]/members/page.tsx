"use client";

import { Users, UserPlus, Shield, Crown, User } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.1 },
  },
};
const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number] },
  },
};

const roleConfig: Record<number, { label: string; icon: typeof Crown; cls: string }> = {
  1: { label: "创建者", icon: Crown, cls: "text-amber-400 bg-amber-500/10" },
  2: { label: "管理员", icon: Shield, cls: "text-blue-400 bg-blue-500/10" },
  3: { label: "成员", icon: User, cls: "text-muted-foreground bg-muted/50" },
};

export default function MembersPage() {
  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible">
      {/* 标题 */}
      <motion.div variants={itemVariants} className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <Users className="h-5 w-5 text-primary" />
          项目成员
        </h2>
        <button
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium",
            "bg-primary text-primary-foreground",
            "hover:opacity-90 active:scale-[0.98] transition-all"
          )}
        >
          <UserPlus className="h-4 w-4" />
          添加成员
        </button>
      </motion.div>

      {/* 角色说明 */}
      <motion.div variants={itemVariants} className="flex items-center gap-4 mb-6">
        {Object.entries(roleConfig).map(([key, config]) => {
          const Icon = config.icon;
          return (
            <div key={key} className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <div className={cn("p-1 rounded", config.cls)}>
                <Icon className="h-3 w-3" />
              </div>
              {config.label}
            </div>
          );
        })}
      </motion.div>

      {/* 空状态 */}
      <motion.div variants={itemVariants}>
        <div className={cn(
          "rounded-xl border border-dashed border-border/40 p-16",
          "flex flex-col items-center justify-center text-center",
          "bg-card/20"
        )}>
          <Users className="h-12 w-12 text-muted-foreground/30 mb-4" />
          <p className="text-lg font-medium mb-1">成员管理</p>
          <p className="text-muted-foreground text-sm">
            邀请团队成员协作，共同完成视频创作项目
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}
