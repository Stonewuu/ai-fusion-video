"use client";

import { useState } from "react";
import { X, Sparkles, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { scriptApi } from "@/lib/api/script";

interface ParseScriptDialogProps {
  open: boolean;
  projectId: number;
  projectName?: string;
  /** "create" = 首次创建, "reparse" = 重新解析（需先删旧剧本） */
  mode?: "create" | "reparse";
  /** 重新解析模式下需传入旧剧本ID，由调用方负责删除 */
  onClose: () => void;
  /** 创建成功后回调，传入剧本信息 */
  onCreated: (script: { id: number; title: string }) => void;
}

export function ParseScriptDialog({
  open,
  projectId,
  projectName,
  mode = "create",
  onClose,
  onCreated,
}: ParseScriptDialogProps) {
  const [title, setTitle] = useState("");
  const [rawContent, setRawContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    const resolvedTitle = title.trim() || projectName?.trim() || "未命名项目";

    if (!rawContent.trim()) {
      setError("请粘贴剧本原文");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const script = await scriptApi.create({
        projectId,
        title: resolvedTitle,
        rawContent: rawContent.trim(),
      });
      const createdTitle = resolvedTitle;
      setTitle("");
      setRawContent("");
      onCreated({ id: script.id, title: createdTitle });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建失败，请重试");
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    if (loading) return;
    onClose();
  };

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
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-purple-400" />
                  <h2 className="text-lg font-semibold">
                    {mode === "reparse" ? "重新解析剧本" : "AI 解析剧本"}
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
                {/* 剧本标题 */}
                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    剧本标题
                  </label>
                  <input
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder={projectName?.trim() ? `留空则使用：${projectName.trim()}` : "留空则使用项目名"}
                    className={cn(
                      "w-full px-3.5 py-2.5 rounded-xl text-sm",
                      "bg-muted/50 border border-border/40",
                      "focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50",
                      "placeholder:text-muted-foreground/50 transition-all"
                    )}
                    autoFocus
                    disabled={loading}
                  />
                </div>

                {/* 剧本原文 */}
                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    剧本原文 <span className="text-destructive">*</span>
                  </label>
                  <textarea
                    value={rawContent}
                    onChange={(e) => setRawContent(e.target.value)}
                    placeholder="在此粘贴完整的剧本原文，AI 将自动解析为结构化的分集、场次和对白数据..."
                    rows={10}
                    className={cn(
                      "w-full px-3.5 py-2.5 rounded-xl text-sm resize-none",
                      "bg-muted/50 border border-border/40",
                      "focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50",
                      "placeholder:text-muted-foreground/50 transition-all"
                    )}
                    disabled={loading}
                  />
                </div>

                {error && <p className="text-sm text-destructive">{error}</p>}
              </div>

              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={handleClose}
                  disabled={loading}
                  className="px-4 py-2 rounded-xl text-sm font-medium hover:bg-muted transition-colors disabled:opacity-50"
                >
                  取消
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={loading}
                  className={cn(
                    "flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-medium",
                    "bg-linear-to-r from-purple-600 to-pink-600",
                    "text-white shadow-lg shadow-purple-500/20",
                    "hover:shadow-purple-500/30 active:scale-[0.98] transition-all",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      解析中...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4" />
                      开始解析
                    </>
                  )}
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
