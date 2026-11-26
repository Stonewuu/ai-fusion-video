import type { ReactNode } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SheetClose } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

interface DirectoryHeaderProps {
  title: string;
  badge?: ReactNode;
  description?: string;
  collapsed?: boolean;
  className?: string;
  children: ReactNode;
}

export function DirectoryHeader({
  title,
  badge,
  description,
  collapsed = false,
  className,
  children,
}: DirectoryHeaderProps) {
  return (
    <header className={cn("shrink-0", className)}>
      <div
        className={cn(
          "flex min-h-9 min-w-0 items-center gap-2",
          collapsed ? "justify-center" : "justify-between",
        )}
      >
        {!collapsed && (
          <div className="flex min-w-0 items-center gap-2">
            <h3 className="truncate text-sm font-semibold leading-5">{title}</h3>
            {badge}
          </div>
        )}
        <div className="flex shrink-0 items-center gap-2">{children}</div>
      </div>
      {!collapsed && description && (
        <p className="mt-0.5 text-xs leading-4 text-muted-foreground">{description}</p>
      )}
    </header>
  );
}

export function DirectorySheetClose({ label }: { label: string }) {
  return (
    <SheetClose
      render={<Button type="button" variant="ghost" size="icon" aria-label={label} title={label} />}
    >
      <X aria-hidden="true" />
    </SheetClose>
  );
}
