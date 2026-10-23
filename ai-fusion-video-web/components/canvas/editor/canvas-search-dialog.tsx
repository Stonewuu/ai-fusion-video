"use client";

import { useEffect, useMemo, useState } from "react";
import { LocateFixed, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";

const PAGE_SIZE = 50;

export function CanvasSearchDialog({
  onClose,
  onInspect,
}: {
  onClose: () => void;
  onInspect: (id: string) => void;
}) {
  const store = useCanvasStoreApi();
  const nodes = useCanvas((state) => state.snapshot!.graph.nodes);
  const catalog = useCanvas((state) => state.catalog!);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [limit, setLimit] = useState(PAGE_SIZE);
  const results = useMemo(() => {
    const search = query.trim().toLocaleLowerCase();
    return nodes.filter((node) =>
      `${node.title} ${catalog.nodes.find((type) => type.kind === node.kind)?.label}`
        .toLocaleLowerCase()
        .includes(search),
    );
  }, [query, nodes, catalog]);
  const index = Math.min(active, Math.max(0, results.length - 1));
  useEffect(() => {
    document
      .getElementById(`canvas-search-result-${index}`)
      ?.scrollIntoView({ block: "nearest" });
  }, [index, limit, query]);
  const choose = (id: string) => {
    store.getState().focus([id]);
    onClose();
    onInspect(id);
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-lg">
        <DialogHeader className="shrink-0">
          <DialogTitle>查找画布节点</DialogTitle>
          <DialogDescription>
            搜索名称或节点类型，选择结果即可定位并打开编辑。
          </DialogDescription>
        </DialogHeader>
        <div className="relative shrink-0">
          <Search className="pointer-events-none absolute left-3 top-2.5 size-4 text-muted-foreground" />
          <Input
            autoFocus
            className="pl-9"
            placeholder="例如：第一场、生成图片、分镜"
            aria-label="搜索画布节点"
            role="combobox"
            aria-expanded
            aria-controls="canvas-search-results"
            aria-activedescendant={
              results[index] ? `canvas-search-result-${index}` : undefined
            }
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setActive(0);
              setLimit(PAGE_SIZE);
            }}
            onKeyDown={(event) => {
              if (event.nativeEvent.isComposing) return;
              if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                event.preventDefault();
                const next = Math.max(
                  0,
                  Math.min(
                    results.length - 1,
                    index + (event.key === "ArrowDown" ? 1 : -1),
                  ),
                );
                setActive(next);
                setLimit((current) =>
                  Math.max(
                    current,
                    Math.ceil((next + 1) / PAGE_SIZE) * PAGE_SIZE,
                  ),
                );
              } else if (event.key === "Enter" && results[index]) {
                event.preventDefault();
                choose(results[index].id);
              }
            }}
          />
        </div>
        <p role="status" className="shrink-0 text-xs text-muted-foreground">
          {results.length > limit
            ? `显示 ${limit} / ${results.length} 个节点`
            : `${results.length} 个节点`}
        </p>
        <OverlayScrollArea className="min-h-0 flex-1">
          <div
            id="canvas-search-results"
            role="listbox"
            aria-label="匹配节点"
            className="space-y-2 p-1"
          >
            {results.slice(0, limit).map((node, i) => (
              <Button
                key={node.id}
                id={`canvas-search-result-${i}`}
                role="option"
                tabIndex={-1}
                aria-selected={index === i}
                aria-posinset={i + 1}
                aria-setsize={results.length}
                variant={index === i ? "secondary" : "ghost"}
                className="w-full justify-start"
                onPointerMove={() => setActive(i)}
                onClick={() => choose(node.id)}
              >
                <LocateFixed />
                <span className="min-w-0 flex-1 truncate text-left">
                  {node.title}
                </span>
                <span className="text-xs text-muted-foreground">
                  {catalog.nodes.find((type) => type.kind === node.kind)?.label}
                </span>
              </Button>
            ))}
            {!results.length && (
              <p className="py-4 text-sm text-muted-foreground">
                {nodes.length
                  ? "没有匹配的节点，试试其他名称。"
                  : "画布还是空的，先添加一个节点。"}
              </p>
            )}
          </div>
        </OverlayScrollArea>
        {results.length > limit && (
          <div className="flex shrink-0 justify-end">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setLimit((current) => current + PAGE_SIZE)}
            >
              显示更多
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
