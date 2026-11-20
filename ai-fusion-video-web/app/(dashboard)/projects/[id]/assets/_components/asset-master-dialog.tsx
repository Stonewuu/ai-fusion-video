"use client";

import type { ReactNode, RefObject } from "react";
import { Check, Loader2, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";

interface Props {
  mode: "create" | "edit";
  description: string;
  saving: boolean;
  disabled: boolean;
  canSubmit: boolean;
  nameRef: RefObject<HTMLInputElement | null>;
  onCancel: () => void;
  onSubmit: () => Promise<void>;
  children: ReactNode;
}

export function AssetMasterDialogContent({
  mode,
  description,
  saving,
  disabled,
  canSubmit,
  nameRef,
  onCancel,
  onSubmit,
  children,
}: Props) {
  const blocked = saving || disabled;
  return (
    <DialogContent
      showCloseButton={!blocked}
      initialFocus={nameRef}
      finalFocus
      style={{ backgroundColor: "var(--background)" }}
      className="flex w-[calc(100%-2rem)] max-h-[calc(100dvh-2rem)] flex-col gap-0 overflow-hidden rounded-2xl p-0 duration-200 data-closed:duration-150 motion-reduce:animate-none sm:max-w-xl"
    >
      <DialogHeader className="shrink-0 gap-2 border-b border-border/20 px-5 py-5 pr-14 sm:px-6 sm:pr-14">
        <DialogTitle className="text-lg font-semibold">
          {mode === "create" ? "新建主资产" : "编辑主资产"}
        </DialogTitle>
        <DialogDescription className="line-clamp-2 break-words">
          {description}
        </DialogDescription>
      </DialogHeader>
      <form
        className="flex min-h-0 flex-col"
        noValidate
        onSubmit={(event) => {
          event.preventDefault();
          if (!blocked) {
            void onSubmit();
          }
        }}
      >
        <DialogScrollBody className="px-5 py-5 sm:px-6">
          {children}
        </DialogScrollBody>
        <DialogFooter className="shrink-0 flex-row items-center justify-end gap-2 border-t border-border/20 bg-muted/20 px-5 py-4 sm:px-6">
          <Button
            type="button"
            variant="outline"
            disabled={blocked}
            onClick={onCancel}
          >
            取消
          </Button>
          <Button type="submit" disabled={blocked || !canSubmit}>
            {saving ? (
              <Loader2 className="animate-spin" />
            ) : mode === "create" ? (
              <Plus />
            ) : (
              <Check />
            )}
            {saving
              ? mode === "create"
                ? "创建中"
                : "保存中"
              : mode === "create"
                ? "创建"
                : "保存"}
          </Button>
        </DialogFooter>
      </form>
    </DialogContent>
  );
}
