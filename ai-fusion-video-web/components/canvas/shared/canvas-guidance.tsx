import type { ReactNode } from "react";
import { CircleAlert, CircleCheck, Info, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const tones = {
  info: { Icon: Info, color: "text-sky-700 dark:text-sky-300", surface: "border-sky-500/20 bg-sky-500/5" },
  action: { Icon: Sparkles, color: "text-violet-600 dark:text-violet-300", surface: "border-violet-500/20 bg-violet-500/5" },
  warning: { Icon: CircleAlert, color: "text-amber-700 dark:text-amber-300", surface: "border-amber-500/30 bg-amber-500/5" },
  error: { Icon: CircleAlert, color: "text-destructive", surface: "border-destructive/20 bg-destructive/5" },
  success: { Icon: CircleCheck, color: "text-emerald-700 dark:text-emerald-300", surface: "border-emerald-500/20 bg-emerald-500/5" },
};

export function CanvasGuidance({ title, children, tone = "info", actions, className }: {
  title: string;
  children?: ReactNode;
  tone?: keyof typeof tones;
  actions?: ReactNode;
  className?: string;
}) {
  const { Icon, color, surface } = tones[tone];
  return (
    <div role={tone === "error" ? "alert" : "status"} className={cn("flex items-start gap-3 rounded-lg border p-3", surface, className)}>
      <Icon aria-hidden className={cn("mt-0.5 size-4 shrink-0", color)} />
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <p className={cn("text-sm font-medium", color)}>{title}</p>
        {children && <div className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">{children}</div>}
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
    </div>
  );
}
