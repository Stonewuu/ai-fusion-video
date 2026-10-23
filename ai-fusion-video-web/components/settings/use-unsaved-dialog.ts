"use client";

import { useEffect } from "react";
import { useConfirm } from "@/components/ui/confirm-dialog";

export function useUnsavedDialog(open: boolean, dirty: boolean, saving: boolean, onOpenChange: (open: boolean) => void) {
  const { confirm } = useConfirm();
  useEffect(() => {
    if (!open || !dirty) return;
    const guard = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener("beforeunload", guard);
    return () => window.removeEventListener("beforeunload", guard);
  }, [open, dirty]);
  return async (next: boolean) => {
    if (saving) return;
    if (!next && dirty && !await confirm({ title: "放弃未保存的配置？", description: "当前修改尚未保存。", confirmText: "放弃修改", cancelText: "继续编辑", variant: "warning" })) return;
    onOpenChange(next);
  };
}
