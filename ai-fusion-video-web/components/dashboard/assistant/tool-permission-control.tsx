"use client";

import {
  CircleCheckBig,
  LockKeyholeOpen,
  MessageCircleQuestion,
  ShieldCheck,
  type LucideIcon,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ToolExecutionMode } from "@/lib/api/ai-assistant";
import { cn } from "@/lib/utils";

const MODE_OPTIONS: Array<{
  value: ToolExecutionMode;
  label: string;
  description: string;
  icon: LucideIcon;
  iconColor: string;
  iconSurfaceClassName: string;
}> = [
  {
    value: "DEFAULT",
    label: "默认",
    description: "读取自动执行，编辑和高风险操作需要批准",
    icon: ShieldCheck,
    iconColor: "#0ea5e9",
    iconSurfaceClassName: "bg-sky-500/10",
  },
  {
    value: "ALWAYS_ASK",
    label: "总是询问",
    description: "每次工具调用都需要你的同意",
    icon: MessageCircleQuestion,
    iconColor: "#f59e0b",
    iconSurfaceClassName: "bg-amber-500/10",
  },
  {
    value: "ALWAYS_ALLOW",
    label: "总是允许",
    description: "读取和编辑自动执行，高风险操作仍需批准",
    icon: CircleCheckBig,
    iconColor: "#10b981",
    iconSurfaceClassName: "bg-emerald-500/10",
  },
  {
    value: "FULL_ACCESS",
    label: "完全访问",
    description: "所有工具（包括高风险操作）均自动执行",
    icon: LockKeyholeOpen,
    iconColor: "#f43f5e",
    iconSurfaceClassName: "bg-rose-500/10",
  },
];

const SELECT_ITEMS = MODE_OPTIONS.map(({ value, label }) => ({ value, label }));

export function AssistantToolPermissionControl({
  value,
  disabled,
  onChange,
}: {
  value: ToolExecutionMode;
  disabled: boolean;
  onChange: (value: ToolExecutionMode) => void;
}) {
  const active = MODE_OPTIONS.find((option) => option.value === value);
  if (!active) {
    throw new Error(`Unsupported tool execution mode: ${value}`);
  }
  const ActiveIcon = active.icon;

  return (
    <Select
      value={value}
      onValueChange={(next) => {
        const selected = MODE_OPTIONS.find((option) => option.value === next);
        if (!selected) {
          throw new Error(`Unsupported tool execution mode: ${String(next)}`);
        }
        onChange(selected.value);
      }}
      items={SELECT_ITEMS}
      disabled={disabled}
    >
      <SelectTrigger
        size="sm"
        className="max-w-36 border-0 bg-transparent px-2 text-xs shadow-none focus-visible:ring-0"
        aria-label={`工具执行方式：${active.label}`}
        data-assistant-interactive="true"
      >
        <SelectValue className="sr-only" placeholder="工具权限" />
        <ActiveIcon className="size-4" style={{ color: active.iconColor }} aria-hidden="true" />
        <span className="min-w-0 flex-1 truncate text-left text-[11px] font-medium">
          {active.label}
        </span>
      </SelectTrigger>
      <SelectContent className="min-w-72 text-xs">
        <SelectGroup>
          {MODE_OPTIONS.map((option) => {
            const OptionIcon = option.icon;
            return (
              <SelectItem
                key={option.value}
                value={option.value}
                className="min-h-12 py-2 text-xs"
              >
                <span
                  className={cn(
                    "grid size-7 shrink-0 place-items-center rounded-lg",
                    option.iconSurfaceClassName,
                  )}
                >
                  <OptionIcon
                    className="size-4"
                    style={{ color: option.iconColor }}
                    aria-hidden="true"
                  />
                </span>
                <span className="min-w-0 flex-1 leading-tight">
                  <span className="block truncate text-xs font-medium">{option.label}</span>
                  <span className="mt-0.5 block whitespace-normal text-[10px] leading-4 text-muted-foreground">
                    {option.description}
                  </span>
                </span>
              </SelectItem>
            );
          })}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
