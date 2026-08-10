"use client";

import { useState } from "react";
import { X, Sparkles, Loader2, AlertTriangle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import type { ScriptEpisode } from "@/lib/api/script";
import { Checkbox } from "@/components/ui/checkbox";

interface EpisodeParseDialogProps {
  open: boolean;
  episode: ScriptEpisode | null;
  /** 该集是否已有场次数据 */
  hasExistingScenes: boolean;
  existingSceneCount?: number;
  onClose: () => void;
  /** 用户确认后回调，传入粘贴的剧本原文 */
  onStartParse: (rawContent: string) => void;
}

export function EpisodeParseDialog({
  open,
  episode,
  hasExistingScenes,
  existingSceneCount = 0,
  onClose,
  onStartParse,
}: EpisodeParseDialogProps) {
  const [rawContent, setRawContent] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = () => {
    if (!rawContent.trim()) {
      setError("请粘贴该集的剧本原文");
      return;
    }
    if (hasExistingScenes && !confirmed) {
      setError("请先确认覆盖现有数据");
      return;
    }
    setError("");
    onStartParse(rawContent.trim());
    // 重置状态
    setRawContent("");
    setConfirmed(false);
    onClose();
  };

  const handleClose = () => {
    setRawContent("");
    setConfirmed(false);
    setError("");
    onClose();
  };

  if (!episode) return null;

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="modal-overlay fixed inset-0 z-50"
            onClick={handleClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2 }}
            className="fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg"
          >
            <div className="modal-surface rounded-2xl border border-border/40 p-6">
              {/* 标题栏 */}
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-purple-400" />
                  <h2 className="text-lg font-semibold">
                    AI 解析 · 第 {episode.episodeNumber} 集
                  </h2>
                </div>
                <button
                  onClick={handleClose}
                  className="p-1.5 rounded-lg hover:bg-muted transition-colors"
                >
                  <X className="h-4 w-4 text-muted-foreground" />
                </button>
              </div>

              <div className="space-y-4">
                {/* 覆盖警告 */}
                {hasExistingScenes && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={cn(
                      "flex items-start gap-3 p-3.5 rounded-xl",
                      "bg-amber-500/10 border border-amber-500/20"
                    )}
                  >
                    <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-amber-800 dark:text-amber-200 font-semibold">
                        该集已有 {existingSceneCount} 个场次
                      </p>
                      <p className="text-xs text-amber-700 dark:text-amber-300/80 mt-0.5">
                        AI 解析将覆盖现有的所有场次和对白数据
                      </p>
                      <label className="flex items-center gap-2.5 mt-2.5 cursor-pointer select-none">
                        <Checkbox
                          checked={confirmed}
                          onCheckedChange={(checked) => setConfirmed(!!checked)}
                          className="border-amber-500/40 data-checked:border-amber-600 data-checked:bg-amber-600 dark:data-checked:bg-amber-500 dark:border-amber-500/30"
                        />
                        <span className="text-xs text-amber-900 dark:text-amber-200/90 font-medium">
                          我已了解，确认覆盖现有数据
                        </span>
                      </label>
                    </div>
                  </motion.div>
                )}

                {/* 剧本原文输入 */}
                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    该集剧本原文 <span className="text-destructive">*</span>
                  </label>
                  <textarea
                    value={rawContent}
                    onChange={(e) => setRawContent(e.target.value)}
                    placeholder="在此粘贴该集的剧本原文，AI 将解析为结构化的场次和对白数据……"
                    rows={10}
                    className={cn(
                      "w-full px-3.5 py-2.5 rounded-xl text-sm resize-none",
                      "bg-muted/50 border border-border/40",
                      "focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50",
                      "placeholder:text-muted-foreground/50 transition-all"
                    )}
                    autoFocus
                  />
                </div>

                {error && <p className="text-sm text-destructive">{error}</p>}
              </div>

              {/* 按钮区 */}
              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={handleClose}
                  className="px-4 py-2 rounded-xl text-sm font-medium hover:bg-muted transition-colors"
                >
                  取消
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={hasExistingScenes && !confirmed}
                  className={cn(
                    "flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-medium",
                    "bg-linear-to-r from-purple-600 to-pink-600",
                    "text-white shadow-lg shadow-purple-500/20",
                    "hover:shadow-purple-500/30 active:scale-[0.98] transition-all",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                >
                  <Sparkles className="h-4 w-4" />
                  开始解析
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
