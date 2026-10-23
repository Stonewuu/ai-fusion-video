"use client";

import { useMemo, useState } from "react";
import { LocateFixed } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";

const PAGE_SIZE = 50;

export function CanvasPlacedNodes({ search }: { search: string }) {
  const store = useCanvasStoreApi();
  const nodes = useCanvas((state) => state.snapshot!.graph.nodes);
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState({ search, limit: PAGE_SIZE });
  const limit = page.search === search ? page.limit : PAGE_SIZE;
  const matching = useMemo(() => {
    const query = search.trim().toLowerCase();
    return nodes.filter((node) => node.title.toLowerCase().includes(query));
  }, [nodes, search]);

  return (
    <details
      className="border-t border-border/20 pt-4"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer text-sm font-semibold outline-none focus-visible:ring-2 focus-visible:ring-ring">
        已放入画布 · {nodes.length}
      </summary>
      {open && (
        <div
          className="mt-3 space-y-2"
          role="region"
          aria-label="已放入画布的节点"
        >
          {matching.slice(0, limit).map((node) => (
            <Button
              key={node.id}
              variant="ghost"
              size="sm"
              className="w-full justify-start"
              onClick={() => store.getState().focus([node.id])}
            >
              <LocateFixed />
              <span className="truncate">{node.title}</span>
            </Button>
          ))}
          {matching.length === 0 && (
            <p className="text-sm text-muted-foreground">没有匹配的节点。</p>
          )}
          {matching.length > limit && (
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs text-muted-foreground">
                已显示 {limit} / {matching.length} 个节点
              </p>
              <Button
                variant="outline"
                size="xs"
                onClick={() => setPage({ search, limit: limit + PAGE_SIZE })}
              >
                显示更多
              </Button>
            </div>
          )}
        </div>
      )}
    </details>
  );
}
