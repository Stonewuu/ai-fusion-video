"use client";

import type { ComponentProps, ReactNode } from "react";
import { DirectorySheetClose } from "@/components/shared/directory-header";
import {
  Sheet,
  SheetContent,
  SheetTitle,
} from "@/components/ui/sheet";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { EpisodeTree } from "./episode-tree";

interface SheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ScriptDirectorySheet({
  open,
  onOpenChange,
  treeProps,
}: SheetProps & { treeProps: ComponentProps<typeof EpisodeTree> }) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="left"
        showCloseButton={false}
        className="gap-0 p-0 data-[side=left]:w-[calc(100%-2rem)] motion-reduce:transition-none motion-reduce:transform-none"
      >
        <SheetTitle className="sr-only">剧本目录</SheetTitle>
        <EpisodeTree
          {...treeProps}
          headerAction={<DirectorySheetClose label="关闭剧本目录" />}
        />
      </SheetContent>
    </Sheet>
  );
}

export function ScriptDetailsSheet({
  open,
  onOpenChange,
  title,
  children,
}: SheetProps & { title: string; children: ReactNode }) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        showCloseButton={false}
        className="gap-0 p-0 data-[side=right]:w-[calc(100%-2rem)] motion-reduce:transition-none motion-reduce:transform-none"
      >
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border/20 p-3">
          <SheetTitle>{title}</SheetTitle>
          <DirectorySheetClose label="关闭详情栏" />
        </div>
        <OverlayScrollArea className="min-h-0 flex-1">
          {children}
        </OverlayScrollArea>
      </SheetContent>
    </Sheet>
  );
}
