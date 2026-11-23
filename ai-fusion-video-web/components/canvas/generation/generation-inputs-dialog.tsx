"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import type { CanvasCapability, CanvasNode } from "@/lib/canvas/types";
import { CanvasNodeInputs } from "../inspector/node-inputs";
import { CanvasDialogContent } from "../shared/canvas-dialog";

/** 生成表单的“输入来源”弹窗：管理连线输入，有未保存的参数时可一并保存。 */
export function GenerationInputsDialog({
  open,
  node,
  capability,
  disabled,
  unsaved,
  onOpenChange,
  onSave,
}: {
  open: boolean;
  node: CanvasNode;
  capability: CanvasCapability;
  disabled: boolean;
  unsaved: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <CanvasDialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>输入来源</DialogTitle>
        </DialogHeader>
        <DialogScrollBody>
          <CanvasNodeInputs
            node={node}
            capability={capability}
            disabled={disabled}
            unsaved={unsaved}
            expanded
          />
        </DialogScrollBody>
        {unsaved && (
          <DialogFooter className="shrink-0">
            <Button disabled={disabled} onClick={onSave}>
              保存参数
            </Button>
          </DialogFooter>
        )}
      </CanvasDialogContent>
    </Dialog>
  );
}
