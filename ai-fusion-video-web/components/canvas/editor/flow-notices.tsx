"use client";
import { Button } from "@/components/ui/button";
import type { useCanvasTransfer } from "./use-canvas-transfer";

export function CanvasFlowNotices({
  transfer,
}: {
  transfer: ReturnType<typeof useCanvasTransfer>;
}) {
  return (
    <>
      {(transfer.batch || transfer.error) && (
        <div
          role="status"
          className="pointer-events-auto flex max-w-full flex-wrap items-center gap-2 rounded-2xl border border-border/40 bg-popover/80 p-3 text-sm shadow-xl backdrop-blur-xl"
        >
          <p className="min-w-0 flex-1">
            {transfer.error ??
              `正在导入 ${transfer.batch!.items.filter((item) => item.media).length} / ${transfer.batch!.items.length} 个素材`}
          </p>
          {transfer.batch && !transfer.busy && (
            <Button variant="outline" size="sm" onClick={transfer.retry}>
              重试导入
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={transfer.cancel}>
            {transfer.busy ? "停止导入" : "关闭"}
          </Button>
        </div>
      )}
    </>
  );
}
