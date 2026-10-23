"use client";

import type { ComponentProps } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DialogClose, DialogContent } from "@/components/ui/dialog";

/** 沿用公共浮层与焦点行为，只补充画布的中文关闭名称。 */
export function CanvasDialogContent({ children, showCloseButton = true, ...props }: ComponentProps<typeof DialogContent>) {
  return (
    <DialogContent {...props} showCloseButton={false}>
      {children}
      {showCloseButton && (
        <DialogClose aria-label="关闭对话框" render={<Button variant="ghost" size="icon-sm" className="absolute right-4 top-4" />}>
          <X aria-hidden />
        </DialogClose>
      )}
    </DialogContent>
  );
}
