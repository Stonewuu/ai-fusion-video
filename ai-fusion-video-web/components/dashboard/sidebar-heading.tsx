import { ArrowLeft, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface SidebarHeadingProps {
  title: string;
  description?: string | null;
  collapsed: boolean;
  navigationId: string;
  onBack: (() => void) | null;
  onCollapsedChange?: (collapsed: boolean) => void;
}

export function SidebarHeading({
  title, description, collapsed, navigationId, onBack, onCollapsedChange,
}: SidebarHeadingProps) {
  const toggleLabel = collapsed ? "展开菜单栏" : "收起菜单栏";
  const ToggleIcon = collapsed ? PanelLeftOpen : PanelLeftClose;
  return (
    <div className={cn("flex flex-col gap-2 pb-2", collapsed ? "items-center" : "px-1")}>
      <div className={cn("flex min-h-9 items-center gap-2", collapsed ? "justify-center" : "justify-between")}>
        {!collapsed && (onBack ? (
          <Button type="button" variant="ghost" onClick={onBack}>
            <ArrowLeft aria-hidden="true" />
            <span className="leading-5">返回项目列表</span>
          </Button>
        ) : (
          <h3 className="min-w-0 truncate px-3 text-sm font-semibold leading-5">{title}</h3>
        ))}
        {onCollapsedChange && (
          <Button
            type="button"
            size="icon"
            variant="ghost"
            title={toggleLabel}
            aria-label={toggleLabel}
            aria-expanded={!collapsed}
            aria-controls={navigationId}
            onClick={() => onCollapsedChange(!collapsed)}
          >
            <ToggleIcon aria-hidden="true" />
          </Button>
        )}
      </div>
      {onBack && (collapsed ? (
        <Button type="button" variant="ghost" size="icon" onClick={onBack} title="返回项目列表" aria-label="返回项目列表">
          <ArrowLeft aria-hidden="true" />
        </Button>
      ) : (
        <div className="flex min-w-0 flex-col gap-1 border-t border-border/30 px-3 pt-3 pb-1">
          <h3 className="truncate text-sm font-semibold leading-5" title={title}>项目：{title}</h3>
          {description && <p className="truncate text-xs leading-4 text-muted-foreground" title={description}>{description}</p>}
        </div>
      ))}
    </div>
  );
}
