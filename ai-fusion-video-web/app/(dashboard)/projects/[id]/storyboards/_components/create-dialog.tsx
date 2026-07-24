"use client";

import { useState } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { storyboardApi } from "@/lib/api/storyboard";

export function CreateStoryboardDialog({
  open,
  projectId,
  projectName,
  onClose,
  onCreated,
}: {
  open: boolean;
  projectId: number;
  projectName?: string;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    const resolvedTitle = title.trim() || projectName?.trim() || "未命名项目";

    setLoading(true);
    setError("");
    try {
      await storyboardApi.create({
        projectId,
        title: resolvedTitle,
      });
      setTitle("");
      onCreated();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建失败");
    } finally {
      setLoading(false);
    }
  };

  if (!open) return null;

  return (
    <>
      <div
        className="modal-overlay fixed inset-0 z-50"
        onClick={onClose}
      />
      <div className="fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2 w-full max-w-md">
        <div className="modal-surface rounded-2xl border border-border/40 p-6">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-semibold">创建空白分镜</h2>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-muted transition-colors"
            >
              <X className="h-4 w-4 text-muted-foreground" />
            </button>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1.5">
                分镜标题
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
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              />
            </div>
            {error && <p className="text-sm text-destructive">{error}</p>}
          </div>
          <div className="flex justify-end gap-3 mt-6">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-sm font-medium hover:bg-muted transition-colors"
            >
              取消
            </button>
            <button
              onClick={handleSubmit}
              disabled={loading}
              className={cn(
                "px-5 py-2 rounded-xl text-sm font-medium",
                "bg-primary text-primary-foreground",
                "hover:opacity-90 active:scale-[0.98] transition-all",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              {loading ? "创建中..." : "创建"}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
