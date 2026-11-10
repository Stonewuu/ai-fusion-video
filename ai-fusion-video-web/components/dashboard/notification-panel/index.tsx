"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { Maximize2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import { openExecutionRecord } from "@/lib/execution-record-navigation";
import { NotificationViews, type NotificationView } from "./notification-views";
import { ExpandedPanel } from "./expanded-panel";
import { PipelineTaskCard } from "./task-list-items";
import { Button } from "@/components/ui/button";

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
    setExpandedTaskId,
  } = usePipelineStore();

  const [view, setView] = useState<NotificationView>("tasks");
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
    if (!notificationOpen || panelExpanded) {
      return;
    }

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
          setExpandedTaskId(null);
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
        <h3 className="text-sm font-semibold">运行与通知</h3>
        <div className="flex items-center gap-2">
          {view === "tasks" && completedTasks.length > 0 && (
            <Button variant="ghost" onClick={clearCompleted}>
              清除已完成
            </Button>
          )}
          <Button variant="ghost" size="icon" onClick={() => setPanelExpanded(true)} aria-label="展开任务面板"><Maximize2 /></Button>
          <Button variant="ghost" size="icon" onClick={() => setNotificationOpen(false)} aria-label="关闭任务中心"><X /></Button>
        </div>
      </div>

      <div className="flex h-[60vh] min-h-0 flex-col">
        <NotificationViews compact view={view} onViewChange={setView} selectedRecord={null} onSelectRecord={openExecutionRecord}>
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
        <div className="p-3"><Button variant="outline" className="w-full" onClick={() => {
          setExpandedTaskId(null);
          setPanelExpanded(true);
        }}>查看全部任务</Button></div>
        </NotificationViews>
      </div>
    </motion.div>,
    document.body
  );
}
