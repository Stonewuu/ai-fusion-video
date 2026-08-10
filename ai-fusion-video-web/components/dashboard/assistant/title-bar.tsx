"use client";

import type { PointerEvent as ReactPointerEvent } from "react";
import {
  Maximize2,
  Minimize2,
  PanelRight,
  PanelRightClose,
  X,
  Menu,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useAssistantStore, type AssistantMode } from "@/lib/store/assistant-store";
import { cn } from "@/lib/utils";
import { AssistantBrandIcon } from "./assistant-brand-icon";

interface TitleBarProps {
  mode: AssistantMode;
  canDock: boolean;
  mobileViewport: boolean;
  showLeadingIcon: boolean;
  onToggleDock: () => void;
  onToggleMaximize: () => void;
  onClose: () => void;
  onStartDrag: (event: ReactPointerEvent<HTMLDivElement>) => void;
}

function IconAction({
  label,
  disabled,
  onClick,
  children,
}: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <span
            className="inline-flex"
            tabIndex={disabled ? 0 : undefined}
            aria-label={disabled ? label : undefined}
            data-assistant-interactive="true"
          >
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-label={label}
              disabled={disabled}
              data-assistant-interactive="true"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={onClick}
            >
              {children}
            </Button>
          </span>
        }
      />
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

export function AssistantTitleBar({
  mode,
  canDock,
  mobileViewport,
  showLeadingIcon,
  onToggleDock,
  onToggleMaximize,
  onClose,
  onStartDrag,
}: TitleBarProps) {
  const selectedId = useAssistantStore((state) => state.selectedConversationId);
  const title = useAssistantStore((state) =>
    selectedId ? state.conversationStates[selectedId]?.conversation.title : undefined,
  );
  const isDocked = mode === "docked";
  const isMaximized = mode === "maximized";
  const conversationMenu = (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      aria-label="打开会话列表"
      title="打开会话列表"
      data-assistant-interactive="true"
      className="assistant-narrow-only"
      onPointerDown={(event) => event.stopPropagation()}
      onClick={() => useAssistantStore.getState().setDrawerOpen(true)}
    >
      <Menu className="size-4" />
    </Button>
  );
  const leadingIcon = showLeadingIcon ? (
    <span className="flex size-8 shrink-0 items-center justify-center">
      <AssistantBrandIcon className="size-7" />
    </span>
  ) : <span className="size-8 shrink-0" aria-hidden="true" />;

  return (
    <div
      className="flex h-14 shrink-0 touch-none select-none items-center gap-3 border-b border-border/20 px-3"
      onPointerDown={onStartDrag}
      onDoubleClick={(event) => {
        if ((event.target as HTMLElement).closest("[data-assistant-interactive]")) return;
        onToggleMaximize();
      }}
    >
      {isDocked ? (
        <>
          {conversationMenu}
          {leadingIcon}
        </>
      ) : (
        <>
          {leadingIcon}
          {conversationMenu}
        </>
      )}

      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-semibold">融光助手</p>
        <p
          className="truncate text-[11px] text-muted-foreground"
          title={title || "新对话"}
        >
          {title || "新对话"}
        </p>
      </div>

      <div className="flex shrink-0 items-center gap-2" data-assistant-interactive="true">
        <IconAction
          label={isDocked ? "退出吸附" : canDock ? "吸附到右侧" : "当前宽度不足，无法吸附"}
          disabled={!isDocked && !canDock}
          onClick={onToggleDock}
        >
          {isDocked ? <PanelRightClose /> : <PanelRight />}
        </IconAction>
        <IconAction
          label={isMaximized
            ? mobileViewport
              ? "移动端保持最大化"
              : "恢复窗口"
            : "最大化"}
          disabled={isMaximized && mobileViewport}
          onClick={onToggleMaximize}
        >
          {isMaximized ? <Minimize2 /> : <Maximize2 />}
        </IconAction>
        <IconAction label="收起助手" onClick={onClose}>
          <X />
        </IconAction>
      </div>
    </div>
  );
}

export function DockedExitHint({ canDock }: { canDock: boolean }) {
  return canDock ? null : (
    <span className={cn("text-[10px] text-muted-foreground")}>窗口宽度不足</span>
  );
}
