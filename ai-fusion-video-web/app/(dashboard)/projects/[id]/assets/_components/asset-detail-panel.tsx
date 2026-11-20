"use client";

import { type ReactNode, useState, useSyncExternalStore } from "react";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

const DESKTOP_QUERY = "(min-width: 1024px)";

function subscribeDesktop(onChange: () => void) {
  const query = window.matchMedia(DESKTOP_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

function desktopSnapshot() {
  return window.matchMedia(DESKTOP_QUERY).matches;
}

function serverSnapshot() {
  return false;
}

interface AssetDetailPanelProps {
  open: boolean;
  editing: boolean;
  onOpenChange: (open: boolean) => void;
  children: ReactNode;
}

export function AssetDetailPanel({
  open,
  editing,
  onOpenChange,
  children,
}: AssetDetailPanelProps) {
  const desktop = useSyncExternalStore(
    subscribeDesktop,
    desktopSnapshot,
    serverSnapshot,
  );
  const [inline, setInline] = useState(desktop);

  // 编辑结束后再切换详情的挂载位置，保留草稿和嵌套弹窗的焦点。
  if (!editing && inline !== desktop) {
    setInline(desktop);
  }

  return (
    <>
      <section
        aria-label="资产工作区"
        className="hidden min-h-0 min-w-0 flex-col overflow-hidden rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm lg:flex"
      >
        {inline && children}
      </section>
      <Sheet
        open={open && !inline}
        onOpenChange={(nextOpen, details) => {
          if (editing) {
            details.cancel();
            return;
          }
          onOpenChange(nextOpen);
        }}
      >
        <SheetContent
          side="right"
          showCloseButton={false}
          finalFocus
          style={{ backgroundColor: "var(--background)" }}
          className="gap-0 overflow-hidden border-border/30 p-0 ease-out data-[side=right]:w-full data-[side=right]:sm:max-w-2xl data-ending-style:duration-150 data-ending-style:ease-in motion-reduce:transform-none motion-reduce:transition-none"
        >
          <SheetHeader className="shrink-0 flex-row items-center gap-3 border-b border-border/20 p-4">
            <SheetClose
              disabled={editing}
              render={<Button variant="ghost" />}
            >
              <ArrowLeft />
              返回目录
            </SheetClose>
            <SheetTitle className="min-w-0 flex-1 text-right font-semibold">
              资产详情
            </SheetTitle>
          </SheetHeader>
          <SheetDescription className="sr-only">
            查看主资产及其子资产。
          </SheetDescription>
          {!inline && children}
        </SheetContent>
      </Sheet>
    </>
  );
}
