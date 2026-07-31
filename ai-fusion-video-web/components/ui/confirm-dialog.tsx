"use client";

import * as React from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Sparkles, AlertTriangle, AlertCircle, Info, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";

export type ConfirmVariant = "ai" | "destructive" | "warning" | "info";

export interface ConfirmOptions {
  title: string;
  description?: React.ReactNode;
  note?: React.ReactNode;
  confirmText?: string;
  cancelText?: string;
  variant?: ConfirmVariant | (string & {});
  /** 是否只显示单个确认按钮（用于 alert 提醒场景） */
  alertOnly?: boolean;
}


interface ConfirmContextType {
  confirm: (options: ConfirmOptions) => Promise<boolean>;
  alert: (options: Omit<ConfirmOptions, "alertOnly" | "cancelText">) => Promise<void>;
}

const ConfirmContext = React.createContext<ConfirmContextType | null>(null);

export function ConfirmProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = React.useState<{
    open: boolean;
    options: ConfirmOptions;
    resolve: (value: boolean) => void;
  } | null>(null);

  const confirm = React.useCallback((options: ConfirmOptions): Promise<boolean> => {
    return new Promise<boolean>((resolve) => {
      setState({
        open: true,
        options,
        resolve,
      });
    });
  }, []);

  const alert = React.useCallback((options: Omit<ConfirmOptions, "alertOnly" | "cancelText">): Promise<void> => {
    return new Promise<void>((resolve) => {
      setState({
        open: true,
        options: { ...options, alertOnly: true },
        resolve: () => resolve(),
      });
    });
  }, []);

  const handleClose = (result: boolean) => {
    if (state) {
      state.resolve(result);
      setState(null);
    }
  };

  return (
    <ConfirmContext.Provider value={{ confirm, alert }}>
      {children}
      {state && (
        <ConfirmDialogModal
          open={state.open}
          options={state.options}
          onResponse={handleClose}
        />
      )}
    </ConfirmContext.Provider>
  );
}

export function useConfirm() {
  const ctx = React.useContext(ConfirmContext);
  if (!ctx) {
    throw new Error("useConfirm must be used within a ConfirmProvider");
  }
  return ctx;
}

export function ConfirmDialogModal({
  open,
  options,
  onResponse,
}: {
  open: boolean;
  options: ConfirmOptions;
  onResponse: (result: boolean) => void;
}) {
  const {
    title,
    description,
    note,
    confirmText,
    cancelText = "取消",
    variant = "info",
    alertOnly = false,
  } = options;

  const renderIcon = () => {
    switch (variant) {
      case "ai":
        return <Sparkles className="h-5 w-5 text-violet-500 shrink-0" />;
      case "destructive":
        return <AlertTriangle className="h-5 w-5 text-destructive shrink-0" />;
      case "warning":
        return <AlertCircle className="h-5 w-5 text-amber-500 shrink-0" />;
      case "info":
      default:
        return <Info className="h-5 w-5 text-primary shrink-0" />;
    }
  };

  const defaultConfirmText = () => {
    if (confirmText) return confirmText;
    if (alertOnly) return "我知道了";
    switch (variant) {
      case "ai":
        return "开始处理";
      case "destructive":
        return "确定删除";
      case "warning":
        return "确定继续";
      case "info":
      default:
        return "确定";
    }
  };

  const buttonVariant = () => {
    switch (variant) {
      case "ai":
        return "ai" as const;
      case "destructive":
        return "destructive" as const;
      case "warning":
      case "info":
      default:
        return "default" as const;
    }
  };

  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) onResponse(false); }}>
      <DialogContent className="sm:max-w-md p-6">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2.5 text-base font-semibold">
            {renderIcon()}
            <span>{title}</span>
          </DialogTitle>
          {description && (
            <DialogDescription className="text-xs text-muted-foreground pt-1.5 leading-relaxed">
              {description}
            </DialogDescription>
          )}
        </DialogHeader>

        {note && (
          <div
            className={cn(
              "rounded-xl border p-3 text-xs leading-relaxed space-y-0.5",
              variant === "ai" && "border-violet-500/20 bg-violet-500/8 text-violet-700 dark:text-violet-300",
              variant === "destructive" && "border-destructive/20 bg-destructive/8 text-destructive",
              variant === "warning" && "border-amber-500/20 bg-amber-500/8 text-amber-700 dark:text-amber-300",
              variant === "info" && "border-border/30 bg-muted/20 text-muted-foreground"
            )}
          >
            {note}
          </div>
        )}

        <DialogFooter className="gap-2 pt-2">
          {!alertOnly && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onResponse(false)}
            >
              {cancelText}
            </Button>
          )}
          <Button
            variant={buttonVariant()}
            size="sm"
            onClick={() => onResponse(true)}
          >
            {variant === "ai" && <Sparkles className="h-3.5 w-3.5" />}
            {variant === "destructive" && <Trash2 className="h-3.5 w-3.5" />}
            {defaultConfirmText()}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
