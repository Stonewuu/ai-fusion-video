"use client";

import { BarChart3, Construction } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.45, ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number] },
  },
};

export default function AnalyticsPage() {
  return (
    <motion.div
      className="max-w-[1200px]"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <motion.div variants={itemVariants} className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <BarChart3 className="h-6 w-6 text-purple-400" />
          数据分析
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          项目与创作数据的可视化分析
        </p>
      </motion.div>

      <motion.div
        variants={itemVariants}
        className={cn(
          "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm",
          "flex flex-col items-center justify-center py-24 text-center"
        )}
      >
        <div className="h-16 w-16 rounded-2xl bg-purple-500/10 flex items-center justify-center mb-5">
          <Construction className="h-8 w-8 text-purple-400/60" />
        </div>
        <h2 className="text-lg font-semibold mb-1.5">功能开发中</h2>
        <p className="text-sm text-muted-foreground max-w-sm">
          数据分析模块正在开发中，敬请期待
        </p>
      </motion.div>
    </motion.div>
  );
}
