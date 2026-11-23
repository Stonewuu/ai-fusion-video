"use client";

import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, LocateFixed } from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import { CanvasNodeTypeIcon } from "../shared/canvas-node-icon";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";

const PAGE_SIZE = 50;

export function CanvasPlacedNodes({ search }: { search: string }) {
  const store = useCanvasStoreApi();
  const nodes = useCanvas((state) => state.snapshot!.graph.nodes);
  const catalog = useCanvas((state) => state.catalog!);
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState({ search, limit: PAGE_SIZE });
  const limit = page.search === search ? page.limit : PAGE_SIZE;
  const matching = useMemo(() => {
    const query = search.trim().toLowerCase();
    return nodes.filter((node) => node.title.toLowerCase().includes(query));
  }, [nodes, search]);

  return (
    <div className="border-t border-foreground/[0.07] pt-3 dark:border-white/[0.08]">
      <button
        type="button"
        aria-expanded={open}
        className="flex h-7 w-full items-center gap-1.5 rounded-[8px] px-1.5 text-left text-xs font-medium text-foreground/85 outline-none transition-colors hover:bg-foreground/[0.05] focus-visible:ring-[3px] focus-visible:ring-primary/30 motion-reduce:transition-none"
        onClick={() => setOpen((current) => !current)}
      >
        <span>已放入画布</span>
        <span className="ml-auto rounded-[6px] bg-foreground/[0.05] px-1.5 py-0.5 text-[11px] font-normal tabular-nums text-muted-foreground dark:bg-white/[0.07]">
          {nodes.length}
        </span>
        {open ? (
          <ChevronDown className="size-3.5 text-muted-foreground" />
        ) : (
          <ChevronRight className="size-3.5 text-muted-foreground" />
        )}
      </button>
      {open && (
        <div
          className="mt-1.5 flex flex-col gap-0.5"
          role="region"
          aria-label="已放入画布的节点"
        >
          {matching.slice(0, limit).map((node) => {
            const capability = catalog.nodes.find(
              (item) => item.kind === node.kind,
            );
            return (
              <CanvasButton
                key={node.id}
                variant="ghost"
                size="sm"
                className="group w-full justify-start"
                onClick={() => store.getState().focus([node.id])}
              >
                {capability && (
                  <CanvasNodeTypeIcon
                    capability={capability}
                    aria-hidden
                    style={{ color: canvasNodeColor(node.kind) }}
                  />
                )}
                <span className="min-w-0 flex-1 truncate text-left">
                  {node.title}
                </span>
                <LocateFixed
                  className="size-3.5 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 motion-reduce:transition-none"
                  aria-hidden
                />
              </CanvasButton>
            );
          })}
          {matching.length === 0 && (
            <p className="py-2 text-center text-xs text-muted-foreground">
              没有匹配的节点
            </p>
          )}
          {matching.length > limit && (
            <div className="flex items-center justify-between gap-2 pt-1.5">
              <p className="text-[11px] tabular-nums text-muted-foreground">
                已显示 {limit} / {matching.length}
              </p>
              <CanvasButton
                variant="secondary"
                size="sm"
                onClick={() => setPage({ search, limit: limit + PAGE_SIZE })}
              >
                显示更多
              </CanvasButton>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
