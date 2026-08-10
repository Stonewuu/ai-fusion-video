"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { Maximize2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import {
  ExpandedPanel,
  PipelineTaskCard,
} from "./detail";

interface NotificationPanelProps {
  anchorRef: RefObject<HTMLButtonElement | null>;
}

export function NotificationPanel({ anchorRef }: NotificationPanelProps) {
  const {
    tasks,
    notificationOpen,
    setNotificationOpen,
    clearCompleted,
    panelExpanded,
    setPanelExpanded,
  } = usePipelineStore();

  const [position, setPosition] = useState({ top: 0, right: 0 });
  const panelRef = useRef<HTMLDivElement>(null);

  const completedTasks = tasks.filter((task) => task.status !== "running");
  const runningTasks = tasks.filter((task) => task.status === "running");

  useEffect(() => {
    if (notificationOpen && anchorRef.current && !panelExpanded) {
      const rect = anchorRef.current.getBoundingClientRect();
      setPosition({
        top: rect.bottom + 8,
        right: window.innerWidth - rect.right,
      });
    }
  }, [notificationOpen, anchorRef, panelExpanded]);

  useEffect(() => {
    if (!notificationOpen || panelExpanded) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (
        panelRef.current &&
        !panelRef.current.contains(event.target as Node) &&
        anchorRef.current &&
        !anchorRef.current.contains(event.target as Node)
      ) {
        setNotificationOpen(false);
      }
    };

    const timer = setTimeout(() => {
      document.addEventListener("mousedown", handleClickOutside);
    }, 10);

    return () => {
      clearTimeout(timer);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [notificationOpen, panelExpanded, anchorRef, setNotificationOpen]);

  useEffect(() => {
    if (!notificationOpen) {
      setPanelExpanded(false);
    }
  }, [notificationOpen, setPanelExpanded]);

  if (!notificationOpen) {
    return null;
  }

  if (panelExpanded) {
    return (
      <ExpandedPanel
        onClose={() => {
          setPanelExpanded(false);
          setNotificationOpen(false);
        }}
      />
    );
  }

  return createPortal(
    <motion.div
      ref={panelRef}
      initial={{ opacity: 0, y: -8, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -8, scale: 0.96 }}
      transition={{ duration: 0.15, ease: "easeOut" }}
      className={cn(
        "fixed z-61",
        // mobile: nearly full width with safe insets
        "left-3 right-3 md:left-auto md:right-[var(--panel-right)] md:w-80",
        "rounded-2xl border border-border/40",
        "bg-card/95 backdrop-blur-xl",
        "shadow-2xl shadow-black/20",
        "overflow-hidden"
      )}
      style={{ top: position.top, "--panel-right": `${position.right}px` } as React.CSSProperties}
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/20">
        <h3 className="text-sm font-semibold">AI 任务</h3>
        <div className="flex items-center gap-1">
          {completedTasks.length > 0 && (
            <button
              onClick={clearCompleted}
              className="text-[10px] text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded-lg hover:bg-muted"
            >
              清除已完成
            </button>
          )}
          <button
            onClick={() => setPanelExpanded(true)}
            className="p-1 rounded-lg hover:bg-muted transition-colors"
            title="展开大面板"
          >
            <Maximize2 className="h-3.5 w-3.5 text-muted-foreground" />
          </button>
          <button
            onClick={() => setNotificationOpen(false)}
            className="p-1 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="h-3.5 w-3.5 text-muted-foreground" />
          </button>
        </div>
      </div>

      <div className="max-h-[60vh] overflow-y-auto">
        {tasks.length === 0 ? (
          <div className="py-12 text-center">
            <p className="text-sm text-muted-foreground">暂无 AI 任务</p>
            <p className="text-xs text-muted-foreground/60 mt-1">
              使用 AI 功能后，任务进度会显示在这里
            </p>
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {runningTasks.length > 0 && (
              <div>
                <p className="text-[10px] font-medium text-muted-foreground px-2 py-1 uppercase tracking-wider">
                  运行中 ({runningTasks.length})
                </p>
                <div className="space-y-1.5">
                  <AnimatePresence>
                    {runningTasks.map((task) => (
                      <PipelineTaskCard key={task.id} task={task} />
                    ))}
                  </AnimatePresence>
                </div>
              </div>
            )}
            {completedTasks.length > 0 && (
              <div>
                <p className="text-[10px] font-medium text-muted-foreground px-2 py-1 uppercase tracking-wider">
                  已完成 ({completedTasks.length})
                </p>
                <div className="space-y-1.5">
                  <AnimatePresence>
                    {completedTasks.map((task) => (
                      <PipelineTaskCard key={task.id} task={task} />
                    ))}
                  </AnimatePresence>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </motion.div>,
    document.body
  );
}