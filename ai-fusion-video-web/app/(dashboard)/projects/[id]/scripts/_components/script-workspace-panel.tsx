"use client";

import type { ReactNode } from "react";
import {
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type Side = "left" | "right";

export function ScriptDetailsPanel({ collapsed, children }: { collapsed: boolean; children: ReactNode }) {
  return (
    <aside
      id="script-right-panel"
      aria-label="剧本详情栏"
      aria-hidden={collapsed}
      inert={collapsed}
      className={cn(
        "relative hidden h-full min-h-0 shrink-0 overflow-hidden bg-card/20 transition-[width] duration-200 ease-out motion-reduce:transition-none 2xl:block",
        collapsed ? "w-0" : "w-72",
      )}
    >
      <div className={cn(
        "flex h-full min-h-0 w-72 flex-col border-l border-border/20 transition-[opacity,transform] duration-200 ease-out motion-reduce:transition-none motion-reduce:transform-none",
        collapsed ? "translate-x-3 opacity-0" : "translate-x-0 opacity-100",
      )}>
        {children}
      </div>
    </aside>
  );
}

export function ScriptPanelToggle({
  side,
  collapsed,
  onToggle,
  sheetOpen,
  onOpenSheet,
}: {
  side: Side;
  collapsed: boolean;
  onToggle: () => void;
  sheetOpen: boolean;
  onOpenSheet: () => void;
}) {
  const label = side === "left" ? "剧本目录" : "详情栏";
  const OpenIcon = side === "left" ? PanelLeftOpen : PanelRightOpen;
  const CloseIcon = side === "left" ? PanelLeftClose : PanelRightClose;
  const action = `${collapsed ? "展开" : "收起"}${label}`;
  return (
    <>
      {side === "right" && (
        <Button
          variant="ghost"
          size="icon"
          className="hidden 2xl:inline-flex"
          aria-label={action}
          title={action}
          aria-expanded={!collapsed}
          aria-controls={`script-${side}-panel`}
          onClick={onToggle}
        >
          {collapsed ? <OpenIcon /> : <CloseIcon />}
        </Button>
      )}
      <Button
        variant="ghost"
        size="icon"
        className={side === "left" ? "xl:hidden" : "2xl:hidden"}
        aria-label={`打开${label}`}
        title={`打开${label}`}
        aria-expanded={sheetOpen}
        aria-haspopup="dialog"
        onClick={onOpenSheet}
      >
        <OpenIcon />
      </Button>
    </>
  );
}
